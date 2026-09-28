#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Offline BH SFPU fragment compiler (Python 3.12+, standard library only).

Usage: python3 tools/sfpu.py kernels.sfpu [-o kernels.inc]
       python3 tools/sfpu.py --self-test
       python3 tools/sfpu.py --examples

Input is Python-shaped syntax, never executed. Each @sfpu function produces a
static inline void C++ function. The caller supplies Dst, configuration, enabled
condition codes, synchronization, and fixed-register inputs. Only LRegs 0..7
are allocated; 9/10 are zero/one, and 11..14 are caller-configured constants.
Other LRegs are untouched. preserve reserves writable registers.
templates=('offset',) declares uint32_t C++ template parameters, separate from
vector arguments. Dst offsets accept affine expressions using +, -, and integer
multipliers. Parameters are in 0..8191; instantiated offsets are checked with
static_assert. Template names cannot be assigned or used as vector operands.
offsets=('row',) declares runtime uint32_t Dst-offset arguments, listed in the
function signature alongside vector arguments. They accept the same affine
address expressions but cannot be assigned or used as vectors or immediates.
The caller guarantees each parameter and resulting address is in 0..8191;
runtime checks are not inserted. Dynamic addresses emit through the caller's
tensix_instruction(uint32_t), declared before the generated include. Constant
addresses retain TTI emission. Scalar helper parameters remain unsupported.
Unpreserved input registers may be clobbered after their last use.
replay=True requires a nonempty body of at most 32 compile-time TTI instructions,
with no runtime parameters or scalar assignments. It emits <name>_instruction_count,
including inserted copies/NOPs. The caller owns recording, replay storage, live
LReg inputs, and RWC/configuration effects; the compiler does not emit REPLAY.

Operations: sfpload(offset, fmt="fp32"|"bf16"), sfpstore(value, offset,
fmt=...), sfpadd(a, b), sfpmul(a, b), sfpmad(a, b, c). Addresses are literal
Dst instruction offsets (0..8191), not byte addresses. Loads/stores accept
addrmod=0..3 (default zero); the caller configures the selected address modifier.
addrmods=('step',) declares compile-time AddrMod parameters, following Dst
template parameters in the generated C++ template argument list. These may only
be used as load/store addrmod arguments; instantiations are checked against 0..3.
Named operands are accepted. ZERO names the architectural zero constant.
constants={11: 0x..., ...} declares exact bit patterns supplied by the caller
in LRegs 11..14. Matching sfploadi calls, including those in inlined helpers,
reuse these read-only registers. No setup or synchronization is emitted.
Input bindings cannot overlap this declaration; undeclared registers are not
assumed to contain any particular value.
bits=('pattern',) declares runtime uint32_t parameters for sfploadi and scalar arithmetic.
They follow runtime offsets in the C++ argument list and always emit two
halfword loads through tensix_instruction().
add32(a,b)/sub32(a,b) are explicit scalar uint32_t arithmetic, wrapping modulo
2^32. They accept scalar parameters, uint32_t literals, and scalar results.
Scalar assignments have value semantics and inline as pure C++ expressions;
they are separate from vectors and cannot occur inside SFPU predicates.
Results can feed sfploadi or Dst addresses (whose range remains the caller's
responsibility). Python arithmetic operators retain their existing offset-only
meaning; add32/sub32 do not promise a particular RISC-V instruction.
sfploadi(imm) broadcasts raw 32-bit bits, reusing read-only LRegs 8/9/10 on an
exact bit match, otherwise using one BF16-immediate instruction when
the low 16 bits are zero and two halfword loads otherwise.
sfpmuli(a, imm) uses a raw 16-bit BF16 immediate. SFPMULI is explicit only;
sfpmul with sfploadi is not folded into it. sfpadd(a, b, negate_b=True) subtracts b. sfpiadd(a, imm)
adds a signed 12-bit immediate to raw bits without changing flags.
Ordinary sfpadd with a finite BF16-exact broadcast can fuse to SFPADDI when its
other operand dies at the add and its output binding permits in-place use,
or a single-use constant load can be removed to pay for the necessary copy.
Subtraction and nonfinite immediates are left unchanged.
sfpmad(a,b,c,negate_product=True) computes c-a*b. sfpsetsgn(a,sign=0|1) replaces
the sign bit without changing any other bits, including NaN payloads. sfpabs(a), sfpneg(a),
and sfprecip(a) map to FP32 absolute value, sign flip, and BH approximate
reciprocal. sfpmin(a,b)/sfpmax(a,b) use SFPSWAP semantics; b must be an
architectural constant register. sfpround(a,fmt="bf16",rounding="nearest")
uses SFPSTOCHRND's nearest BF16 conversion (ties upward in magnitude).
Stochastic rounding is intentionally unsupported, despite the hardware opcode's
name. Only the deterministic nearest BF16 conversion is exposed.
sfpsetexp(a, exponent) replaces a's exponent with exponent's low eight bits.
sfpshft2(a, mode="rotate") rotates each eight-lane subvector right by one lane.
Explicit sfpnop() is unsupported. Mandatory instruction gaps belong in compiler
lowering, not source code. A rotate consuming the preceding ADD/MUL/MAD result
gets an SFPNOP for BH's missed dependency. A rotate immediately consuming a
fragment input gets the same conservative entry gap. Other SFPU instructions
provide that gap themselves; consecutive rotates use hardware interlocking.
Load-macro sequences remain unsupported.

setrwc(dst=0) sets Dst RWC and its carry register to a literal 0..15, leaving
SrcA/SrcB/fidelity and bank state untouched. It is a standalone, unconditional
side effect, emitted in source order without a wait. It is not an SFPU cycle
and does not clear pending SFPU dependency tracking. Other SETRWC fields are
not yet exposed; callers still own cross-engine synchronization.

with sfpiadd(test, -1, cc="gte0") (or cc="lt0") enables all lanes, sets the
mask from the signed integer sum, and runs a non-nested region. Alternatively,
with sfpsetcc(test,cc="lt0"|"gte0"|"eq0"|"ne0") tests raw signed bits
against zero; notably negative zero satisfies lt0. Existing names
assigned there retain their prior inactive lanes; new names are region-local.
Inactive-lane seeds are copied before entry, then masked results are copied
into those destinations before exit. A last-use arithmetic update can reuse
the old value's register instead. Stores within the region are masked.
Exit disables predication and clears flags, matching SFPENCC(0,0,0,10).
a,b,c,d = sfptransp(a,b,c,d) transposes four vectors; an eight-vector form
transposes both groups. Lowering places operands in LRegs 0..3 or 0..7 and
clobbers all eight registers. Preserved or live-through values cause an error.
Operands are copied into place as needed; a cyclic shuffle without scratch
space is rejected. Results occupy the corresponding physical registers; an
incompatible output binding is rejected rather than silently changing semantics.
Transpose is unconditional only. Loops, arbitrary Python, and scheduling are unsupported.
No fusion or arithmetic reassociation occurs.
@sfpu_helper() defines a reusable vector-valued helper with plain arguments and
a final return. Calls are hygienically inlined before allocation, including
named arguments and nested expressions. Helpers have no physical register ABI;
only @sfpu entry points bind registers. Helpers can return a vector or a flat
tuple of vectors; tuple assignment evaluates all values before rebinding names.
Recursion is unsupported; helper compile-time parameters are deferred.
Nested expressions emit left-to-right. Stores remain in source order.

Return names must match outputs exactly. Required output registers constrain
allocation directly; no spills are inserted. Input-to-output, tied-operand,
and masked-merge copies are inserted when needed. A feasible block
may therefore fail this deliberately simple allocator. Source line annotations
and the register contract are emitted for auditing. This compiler does not prove
hardware timing safety: callers still own engine hazards and configuration.
"""

import argparse
import ast
import copy
from dataclasses import dataclass, field
from pathlib import Path
import sys
import unittest
import re
import subprocess
import tempfile


class CompileError(Exception):
    pass


@dataclass(eq=False)
class Value:
    op: str
    args: list = field(default_factory=list)
    line: int = 0
    offset: int | str = 0
    fmt: int = 3
    reg: int | None = None
    mod: int = 0


class Compiler:
    def is_scalar(self, node):
        return ((isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                 node.func.id in ("add32", "sub32")) or
                (isinstance(node, ast.Name) and node.id in self.scalars))

    def scalar(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 0xffffffff:
            return f"{node.value}u"
        if isinstance(node, ast.Name):
            if node.id in self.scalars:
                return self.scalars[node.id]
            if node.id in (*self.offsets, *self.bits, *self.templates):
                return node.id
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("add32", "sub32"):
            if len(node.args) != 2 or node.keywords:
                self.fail(node, "add32/sub32 require two positional uint32_t operands")
            a, b = (self.scalar(arg) for arg in node.args)
            operator = "+" if node.func.id == "add32" else "-"
            return f"(uint32_t({a}) {operator} uint32_t({b}))"
        self.fail(node, "expected a uint32_t parameter, literal, or add32/sub32 expression")

    def address(self, node):
        if self.is_scalar(node):
            expr = self.scalar(node)
            self.runtime_offsets.add(expr)
            return expr
        def affine(n):
            if isinstance(n, ast.Name) and n.id in (*self.templates, *self.offsets):
                return {n.id: 1}, 0
            if isinstance(n, ast.Constant) and type(n.value) is int:
                return {}, n.value
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                terms, constant = affine(n.operand)
                sign = -1 if isinstance(n.op, ast.USub) else 1
                return {k: sign * v for k, v in terms.items()}, sign * constant
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
                a, x = affine(n.left)
                b, y = affine(n.right)
                if isinstance(n.op, ast.Mult):
                    if a and b:
                        self.fail(n, "offset multiplication requires a constant operand")
                    return ({k: v * y for k, v in a.items()} if a else
                            {k: v * x for k, v in b.items()}), x * y
                sign = -1 if isinstance(n.op, ast.Sub) else 1
                result = a.copy()
                for k, v in b.items():
                    result[k] = result.get(k, 0) + sign * v
                return result, x + sign * y
            self.fail(n, "expected an affine Dst offset")
        terms, constant = affine(node)
        terms = {k: v for k, v in terms.items() if v}
        if not terms:
            if not 0 <= constant < 8192:
                self.fail(node, "offset must be an integer in 0..8191")
            return constant
        # Bound signed intermediate arithmetic for every legal template parameter.
        if abs(constant) + 8191 * sum(abs(v) for v in terms.values()) >= 2**63 - 1:
            self.fail(node, "offset expression exceeds signed 64-bit arithmetic")
        expr = "(" + " + ".join([f"{constant}LL"] +
                                  [f"{v}LL * int64_t({k})" for k, v in terms.items()]) + ")"
        if set(terms) & set(self.offsets):
            self.runtime_offsets.add(expr)
        else:
            self.offset_checks.add(expr)
        return expr

    def fail(self, node, message):
        raise CompileError(f"{self.filename}:{getattr(node, 'lineno', 1)}: {message}")

    def literal(self, node):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            self.fail(node, "expected a literal")

    def compile(self, source, filename="<input>"):
        self.filename = filename
        try:
            tree = ast.parse(source, filename)
        except SyntaxError as e:
            raise CompileError(f"{filename}:{e.lineno}: {e.msg}") from None
        names = set()
        output = []
        helpers = {}
        for n in ast.walk(tree):
            if isinstance(n, (ast.Name, ast.arg)) and getattr(n, 'id', getattr(n, 'arg', '')).startswith('__sfpu_inline_'):
                self.fail(n, 'reserved compiler identifier prefix')
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef):
                self.fail(fn, "only function definitions are allowed")
            if fn.name in names:
                self.fail(fn, "duplicate function name")
            names.add(fn.name)
            if (len(fn.decorator_list) == 1 and isinstance(fn.decorator_list[0], ast.Call) and
                    isinstance(fn.decorator_list[0].func, ast.Name) and
                    fn.decorator_list[0].func.id == 'sfpu_helper'):
                dec = fn.decorator_list[0]
                a = fn.args
                parameters = [arg.arg for arg in a.args]
                if (dec.args or dec.keywords or a.posonlyargs or a.vararg or a.kwarg or a.kwonlyargs or
                        a.defaults or fn.returns or any(x.annotation for x in a.args) or
                        getattr(fn, 'type_params', []) or not fn.body or
                        not isinstance(fn.body[-1], ast.Return) or fn.body[-1].value is None or
                        len(set(parameters)) != len(parameters) or
                        set(parameters) & {'ZERO'}):
                    self.fail(fn, 'helper requires plain vector arguments and a final value return')
                helpers[fn.name] = fn
        serial = 0

        def expand(statements, stack=()):
            nonlocal serial
            result = []
            for statement in statements:
                prefix = []

                class Calls(ast.NodeTransformer):
                    def visit_Call(visitor, node):
                        nonlocal serial
                        if not isinstance(node.func, ast.Name) or node.func.id not in helpers:
                            node = visitor.generic_visit(node)
                            if (not isinstance(node.func, ast.Name) or node.func.id in ('sfpstore', 'sfptransp', 'sfpsetcc', 'setrwc') or
                                    (node.func.id == 'sfpiadd' and any(k.arg == 'cc' for k in node.keywords))):
                                return node
                            # Materialize preceding sibling expressions before an inlined call's body.
                            serial += 1
                            name = f'__sfpu_inline_{serial}_value'
                            prefix.append(ast.copy_location(ast.Assign(
                                targets=[ast.Name(id=name, ctx=ast.Store())], value=node), node))
                            return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
                        name = node.func.id
                        if name in stack:
                            self.fail(node, 'recursive SFPU helper call')
                        fn = helpers[name]
                        parameters = [a.arg for a in fn.args.args]
                        if len(node.args) > len(parameters):
                            self.fail(node, 'too many helper arguments')
                        arguments = list(zip(parameters, node.args))
                        used = {n for n, _ in arguments}
                        for kw in node.keywords:
                            if kw.arg not in parameters or kw.arg in used:
                                self.fail(node, 'unknown or repeated helper argument')
                            used.add(kw.arg)
                            arguments.append((kw.arg, kw.value))
                        if used != set(parameters):
                            self.fail(node, 'missing helper argument')
                        serial += 1
                        stem = f'__sfpu_inline_{serial}_'
                        for arg, value in arguments:
                            value = visitor.visit(value)
                            prefix.append(ast.copy_location(ast.Assign(
                                targets=[ast.Name(id=stem+arg, ctx=ast.Store())], value=value), node))

                        class Rename(ast.NodeTransformer):
                            def visit_Name(visitor, n):
                                if n.id == 'ZERO':
                                    if isinstance(n.ctx, ast.Store):
                                        self.fail(n, 'architectural constants cannot be rebound')
                                    return n
                                return ast.copy_location(ast.Name(id=stem+n.id, ctx=n.ctx), n)

                            def visit_Call(visitor, n):
                                # Function names are not local vector bindings.
                                n.args = [visitor.visit(a) for a in n.args]
                                for kw in n.keywords:
                                    kw.value = visitor.visit(kw.value)
                                return n

                        body = [Rename().visit(copy.deepcopy(s)) for s in fn.body]
                        expanded = expand(body, (*stack, name))
                        if not isinstance(expanded[-1], ast.Return):
                            self.fail(fn, 'helper requires a final return')
                        prefix.extend(expanded[:-1])
                        return expanded[-1].value

                statement = copy.deepcopy(statement)
                if isinstance(statement, ast.With):
                    statement.items = [Calls().visit(item) for item in statement.items]
                    statement.body = expand(statement.body, stack)
                else:
                    statement = Calls().visit(statement)
                result.extend(prefix)
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1 and
                        isinstance(statement.targets[0], ast.Tuple) and isinstance(statement.value, ast.Tuple)):
                    targets = statement.targets[0].elts
                    values = statement.value.elts
                    if (not targets or len(targets) != len(values) or
                            any(not isinstance(n, ast.Name) for n in targets) or
                            len({n.id for n in targets}) != len(targets)):
                        self.fail(statement, 'tuple assignment requires matching unique names')
                    temporaries = []
                    for value in values:
                        serial += 1
                        name = f'__sfpu_inline_{serial}_tuple'
                        temporaries.append(name)
                        result.append(ast.copy_location(ast.Assign(
                            targets=[ast.Name(id=name, ctx=ast.Store())], value=value), statement))
                    for target, name in zip(targets, temporaries):
                        result.append(ast.copy_location(ast.Assign(
                            targets=[target], value=ast.Name(id=name, ctx=ast.Load())), statement))
                    continue
                result.append(statement)
            return result

        for fn in tree.body:
            if fn.name in helpers:
                continue
            fn.body = expand(fn.body)
            output.append(self.function(ast.fix_missing_locations(fn)))
        return "\n\n".join(output) + "\n"

    def function(self, fn):
        if (len(fn.decorator_list) != 1 or not isinstance(fn.decorator_list[0], ast.Call)):
            self.fail(fn, "expected one @sfpu(...) decorator")
        dec = fn.decorator_list[0]
        if not isinstance(dec.func, ast.Name) or dec.func.id != "sfpu" or dec.args:
            self.fail(dec, "expected @sfpu with named contract fields")
        contract = {}
        for kw in dec.keywords:
            if kw.arg not in ("inputs", "outputs", "preserve", "templates", "offsets", "addrmods", "bits", "constants", "replay") or kw.arg in contract:
                self.fail(kw, "unknown or repeated contract field")
            contract[kw.arg] = self.literal(kw.value)
            if kw.arg == "constants" and isinstance(kw.value, ast.Dict) and len(contract[kw.arg]) != len(kw.value.keys):
                self.fail(kw, "repeated constant register")
        inputs = contract.get("inputs", {})
        outputs = contract.get("outputs", {})
        preserve = contract.get("preserve", ())
        replay = contract.get("replay", False)
        if type(replay) is not bool:
            self.fail(dec, "replay must be a boolean")
        constants = contract.get("constants", {})
        if not isinstance(constants, dict) or any(type(r) is not int or r not in range(11, 15) or
                                                type(v) is not int or not 0 <= v <= 0xffffffff
                                                for r, v in constants.items()):
            self.fail(dec, "constants must map LRegs 11..14 to uint32_t bit patterns")
        self.constant_values = {8: 0x3f56594b, 9: 0, 10: 0x3f800000, **constants}
        self.templates = contract.get("templates", ())
        self.offsets = contract.get("offsets", ())
        self.addrmods = contract.get("addrmods", ())
        self.bits = contract.get("bits", ())
        for kind, names in (("templates", self.templates), ("offsets", self.offsets),
                            ("addrmods", self.addrmods), ("bits", self.bits)):
            if (not isinstance(names, (tuple, list)) or
                    any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n)
                        for n in names) or len(set(names)) != len(names)):
                self.fail(dec, f"{kind} must list unique identifiers")
        parameters = (*self.templates, *self.offsets, *self.addrmods, *self.bits)
        runtime = (*self.offsets, *self.bits)
        if replay and runtime:
            self.fail(dec, "replay bodies cannot have runtime parameters")
        if len(set(parameters)) != len(parameters):
            self.fail(dec, "parameter names must be distinct")
        self.offset_checks = set()
        self.runtime_offsets = set()
        for mapping, allowed in ((inputs, set(range(8)) | {8, 9, 10, 11, 12, 13, 14}),
                                 (outputs, set(range(8)))):
            if not isinstance(mapping, dict) or any(
                not isinstance(k, str) or type(v) is not int or v not in allowed
                for k, v in mapping.items()
            ):
                self.fail(dec, "invalid register binding")
            if len(set(mapping.values())) != len(mapping):
                self.fail(dec, "aliased register bindings are unsupported")
        if not isinstance(preserve, (tuple, list)) or any(type(r) is not int or r not in range(8) for r in preserve):
            self.fail(dec, "preserve must list writable LRegs 0..7")
        if set(inputs.values()) & set(constants):
            self.fail(dec, "constant register conflicts with an input binding")
        if set(outputs.values()) & set(preserve):
            self.fail(dec, "output register is preserved")
        a = fn.args
        if (a.posonlyargs or a.vararg or a.kwarg or a.kwonlyargs or a.defaults or
                fn.returns or any(x.annotation for x in a.args) or getattr(fn, "type_params", [])):
            self.fail(fn, "only plain positional function arguments are supported")
        if set(x.arg for x in a.args) != set(inputs) | set(runtime) or len(a.args) != len(inputs) + len(runtime):
            self.fail(fn, "every argument must have an input binding or runtime parameter declaration")
        if "ZERO" in inputs:
            self.fail(fn, "ZERO is reserved")
        if set(parameters) & (set(inputs) | set(outputs) | {"ZERO", fn.name}):
            self.fail(fn, "offset parameter conflicts with a vector or function name")
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in parameters:
                self.fail(n, "offset parameters cannot be assigned")
        self.env = {k: Value("input", reg=v) for k, v in inputs.items()}
        self.scalars = {}
        self.env.update(ZERO=Value("input", reg=9))
        self.ops = []
        returned = []
        for i, stmt in enumerate(fn.body):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Tuple):
                target = stmt.targets[0]
                call = stmt.value
                if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or
                        call.func.id != "sfptransp" or call.keywords or len(call.args) not in (4, 8) or
                        len(target.elts) != len(call.args) or
                        any(not isinstance(n, ast.Name) for n in target.elts)):
                    self.fail(stmt, "expected four or eight names assigned from sfptransp")
                names = [n.id for n in target.elts]
                if len(set(names)) != len(names) or set(names) & ({"ZERO"} | set(parameters) | set(self.scalars)):
                    self.fail(stmt, "invalid transpose result names")
                args = [self.expr(n) for n in call.args]
                op = Value("transpose", args, stmt.lineno)
                self.ops.append(op)
                for reg, name in enumerate(names):
                    result = Value("transpose_result", [op], stmt.lineno, reg=reg)
                    self.ops.append(result)
                    self.env[name] = result
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                if name == "ZERO":
                    self.fail(stmt, "architectural constants cannot be rebound")
                if self.is_scalar(stmt.value):
                    if replay:
                        self.fail(stmt, "replay bodies cannot have scalar assignments")
                    if name in self.env:
                        self.fail(stmt, "cannot replace a vector with a scalar")
                    self.scalars[name] = self.scalar(stmt.value)
                else:
                    if name in self.scalars:
                        self.fail(stmt, "cannot replace a scalar with a vector")
                    self.env[name] = self.expr(stmt.value)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self.call(stmt.value, store=True)
            elif isinstance(stmt, ast.With):
                self.region(stmt)
            elif isinstance(stmt, ast.Return) and i == len(fn.body) - 1:
                nodes = stmt.value.elts if isinstance(stmt.value, ast.Tuple) else ([] if stmt.value is None else [stmt.value])
                if any(not isinstance(n, ast.Name) for n in nodes):
                    self.fail(stmt, "return named values only")
                if [n.id for n in nodes] != list(outputs):
                    self.fail(stmt, "return names/order must match outputs")
                returned = [self.expr(n) for n in nodes]
            else:
                self.fail(stmt, "expected assignment, store, or final return")
        if len(returned) != len(outputs):
            self.fail(fn, "missing declared outputs")
        targets = {}
        for i, (value, reg) in enumerate(zip(returned, outputs.values())):
            if value.op == "input" and value.reg != reg:
                value = Value("copy", [value], line=fn.lineno)
                self.ops.append(value)
                returned[i] = value
            if value in targets and targets[value] != reg:
                self.fail(fn, "one value cannot occupy two output registers")
            targets[value] = reg
            if value.op == "merge":
                targets[value.args[0]] = reg
        last = {}
        for i, op in enumerate(self.ops):
            for value in op.args:
                last[value] = i
        for value in returned:
            last[value] = len(self.ops)
        # SFPADDI ties its input and output. A removed single-use constant load
        # can pay for a copy; otherwise require a dying writable operand.
        uses = {}
        for op in self.ops:
            for value in op.args:
                uses[value] = uses.get(value, 0) + 1
        for value in returned:
            uses[value] = uses.get(value, 0) + 1
        fused_constants = set()
        for i, op in enumerate(self.ops):
            if op.op != "sfpadd" or op.mod:
                continue
            for constant, source in (op.args, op.args[::-1]):
                bits = (constant.offset if constant.op == "sfploadi" and type(constant.offset) is int else
                        {9: 0, 10: 0x3f800000}.get(constant.reg) if constant.op == "input" else None)
                if bits is None or bits & 65535 or (bits & 0x7f800000) == 0x7f800000:
                    continue
                removable = constant.op == "sfploadi" and uses.get(constant) == 1
                if last.get(source) != i and not removable:
                    continue
                if source.op == "input" and (source.reg >= 8 or source.reg in preserve) and not removable:
                    continue
                source_target = targets.get(source, source.reg if source.op == "input" else None)
                if op in targets and source_target != targets[op] and not removable:
                    continue
                op.op, op.args, op.offset = "sfpaddi", [source], bits >> 16
                fused_constants.add(constant)
                break
        used = {value for op in self.ops for value in op.args} | set(returned)
        self.ops = [op for op in self.ops if op not in fused_constants or op in used]
        last = {}
        for i, op in enumerate(self.ops):
            for value in op.args:
                last[value] = i
        for value in returned:
            last[value] = len(self.ops)
        positions = {op: i for i, op in enumerate(self.ops)}
        inplace = {}
        # Reserve the old register through the masked update. Only arithmetic
        # that reads before writing may replace its seed; live aliases keep copies.
        for merge in self.ops:
            if merge.op != "merge":
                continue
            slot, new = merge.args
            old = slot.args[0]
            if (new.op in ("sfpadd", "sfpmul", "sfpmad") and old in new.args and
                    last.get(old, -1) == positions[new] and last[new] == positions[merge] and
                    slot not in targets and new not in targets):
                inplace[slot] = new
        reuse = {}
        live = {v.reg: v for op in self.ops for v in op.args if v.op == "input"}
        for v in returned:
            if v.op == "input":
                live[v.reg] = v
        lines = [f"// inputs={inputs!r}; outputs={outputs!r}; preserve={tuple(preserve)!r}",
                 "// Requires all lanes enabled and unconditional execution; caller owns Dst/configuration and engine waits.",
                 f"static inline void {fn.name}({', '.join('uint32_t ' + n for n in runtime) or 'void'}) {{"]
        if constants:
            lines.insert(1, "// Caller supplies constants: " +
                         ", ".join(f"LReg{r}=0x{v:08x}" for r, v in sorted(constants.items())))
        clobbers = set()
        if self.templates or self.addrmods:
            lines.insert(-1, "template<" + ", ".join(f"uint32_t {n}" for n in (*self.templates, *self.addrmods)) + ">")
            for n in self.templates:
                lines.append(f'    static_assert({n} < 8192u, "Dst template parameter out of range");')
            for n in self.addrmods:
                lines.append(f'    static_assert({n} < 4u, "SFPU AddrMod out of range");')
        for expr in sorted(self.offset_checks):
            lines.append(f'    static_assert({expr} >= 0 && {expr} < 8192, "Dst offset out of range");')
        # Unknown producers at the fragment boundary may be pending MAD results.
        pending = set(inputs.values())
        instruction_count = 0

        def tti(text):
            nonlocal instruction_count
            lines.append("    " + text)
            instruction_count += 1

        for i, op in enumerate(self.ops):
            src = [v.reg for v in op.args]
            # Operands are read before destination write, permitting last-use reuse.
            live = {r: v for r, v in live.items() if last.get(v, -1) > i}
            if op in inplace and src[0] < 8 and src[0] not in preserve:
                op.reg = src[0]
                live[op.reg] = op
                reuse[inplace[op]] = op
                continue
            if op.op == "transpose":
                if any(r < 8 for r in live) or preserve:
                    raise CompileError(f"{self.filename}:{op.line}: transpose clobbers live or preserved LRegs")
                # Parallel copies: do not overwrite an operand still needed by another move.
                moves = {dst: reg for dst, reg in enumerate(src) if dst != reg}
                while moves:
                    ready = next((dst for dst in moves if dst not in moves.values()), None)
                    if ready is None:
                        scratch = next((r for r in range(len(op.args), 8) if r not in src), None)
                        if scratch is None:
                            raise CompileError(f"{self.filename}:{op.line}: transpose operand shuffle needs a scratch LReg")
                        old = next(iter(moves))
                        tti(f"TTI_SFPMOV(0u, {old}u, {scratch}u, 2u);")
                        moves = {d: scratch if s == old else s for d, s in moves.items()}
                        src.append(scratch)
                        continue
                    tti(f"TTI_SFPMOV(0u, {moves.pop(ready)}u, {ready}u, 2u);")
                tti(f"TTI_SFPTRANSP(0u, 0u, 0u, 0u); // line {op.line}")
                clobbers.update(range(8))
                pending = set()
                continue
            if op.op == "transpose_result":
                if op in targets and targets[op] != op.reg:
                    raise CompileError(f"{self.filename}:{op.line}: transpose output binding requires a copy")
                live[op.reg] = op
                continue
            if op.op == "merge":
                op.reg = src[0]
                live[op.reg] = op
            elif op.op not in ("sfpstore", "end", "begin_cc", "setrwc"):
                if op in reuse:
                    slot = reuse[op]
                    assert live.get(slot.reg) is slot
                    del live[slot.reg]
                free = [r for r in range(8) if r not in live and r not in preserve]
                # SETEXP reads its exponent from VD; a preparatory copy must not
                # destroy the separate sign/mantissa operand.
                if op.op == "sfpsetexp" and src[0] != src[1]:
                    free = [r for r in free if r != src[0]]
                preferred = src[0] if op.op in ("sfpaddi", "sfpmuli") and src[0] in free else (free[0] if free else None)
                reg = reuse[op].reg if op in reuse else targets.get(op, preferred)
                if reg not in free:
                    raise CompileError(f"{self.filename}:{op.line}: register allocation failed; live LRegs {sorted(live)}")
                op.reg = reg
                live[reg] = op
                clobbers.add(reg)
            def emit(name, fields):
                nonlocal pending
                if ((name == "SFPSHFT2" and fields[1] in pending) or
                        (name == "SFPSWAP" and pending.intersection(fields[1:3]))):
                    tti("TTI_SFPNOP; // BH missed MAD dependency gap")
                operands = ', '.join(str(x) + 'u' if isinstance(x, int) else x for x in fields)
                if ((name in ("SFPLOAD", "SFPSTORE") and op.offset in self.runtime_offsets) or
                        (name == "SFPLOADI" and isinstance(op.offset, str))):
                    if replay:
                        self.fail(fn, "replay bodies require compile-time instruction operands")
                    lines.append(f"    tensix_instruction(TT_OP_{name}({operands})); // line {op.line}")
                else:
                    tti(f"TTI_{name}({operands}); // line {op.line}")
                pending = ({fields[3]} if name in ("SFPADD", "SFPMUL", "SFPMAD") else
                           {fields[1]} if name in ("SFPADDI", "SFPMULI") else set())

            if op.op == "setrwc":
                # Matrix-unit issue is not a guaranteed SFPU dependency gap.
                tti(f"TTI_SETRWC(0u, 0u, {op.offset}u, 0u, 0u, 4u); // line {op.line}")
                continue
            if op.op == "end":
                emit("SFPENCC", [0, 0, 0, 10])
                continue
            if op.op == "begin":
                emit("SFPENCC", [3, 0, 0, 10])
                emit("SFPIADD", [op.offset & 4095, src[0], op.reg, op.mod])
                continue
            if op.op == "begin_cc":
                emit("SFPENCC", [3, 0, 0, 10])
                emit("SFPSETCC", [0, src[0], 0, op.mod])
                continue
            if op.op in ("copy", "merge"):
                if src[-1] != op.reg:
                    emit("SFPMOV", [0, src[-1], op.reg, 2 if op.op == "copy" else 0])
                continue
            if op.op == "sfploadi":
                if isinstance(op.offset, str):
                    emit("SFPLOADI", [op.reg, 8, f"({op.offset} >> 16)"])
                    lines.append(f"    tensix_instruction(pack_u16({op.offset}, "
                                 f"TT_OP_SFPLOADI({op.reg}u, 10u, 0u) >> 16)); // line {op.line}")
                elif op.offset & 65535:
                    emit("SFPLOADI", [op.reg, 8, op.offset >> 16])
                    emit("SFPLOADI", [op.reg, 10, op.offset & 65535])
                else:
                    emit("SFPLOADI", [op.reg, 0, op.offset >> 16])
                continue
            if op.op in ("sfpaddi", "sfpmuli", "sfpsetexp", "sfpmin", "sfpmax"):
                tied = src[1] if op.op == "sfpsetexp" else src[0]
                if tied != op.reg:
                    emit("SFPMOV", [0, tied, op.reg, 0])
            if op.op == "sfpload":
                fields = [op.reg, op.fmt, op.mod, op.offset]
            elif op.op == "sfpstore":
                fields = [src[0], op.fmt, op.mod, op.offset]
            elif op.op == "sfpadd":
                fields = [10, *src, op.reg, op.mod]
            elif op.op == "sfpmul":
                fields = [*src, 9, op.reg, 0]
            elif op.op == "sfpmad":
                fields = [*src, op.reg, op.mod]
            elif op.op in ("sfpmin", "sfpmax"):
                if src[1] < 8:
                    raise CompileError(f"{self.filename}:{op.line}: min/max bound must be an architectural constant LReg")
                emit("SFPSWAP", [0, src[1], op.reg, 1 if op.op == "sfpmin" else 9])
                continue
            elif op.op in ("sfpabs", "sfpneg", "sfpsetsgn", "sfprecip", "sfpround"):
                instruction, fields = {
                    "sfpabs": ("SFPABS", [0, src[0], op.reg, 1]),
                    "sfpneg": ("SFPMOV", [0, src[0], op.reg, 1]),
                    "sfpsetsgn": ("SFPSETSGN", [op.offset, src[0], op.reg, 1]),
                    "sfprecip": ("SFPARECIP", [0, src[0], op.reg, 0]),
                    "sfpround": ("SFP_STOCH_RND", [0, 0, src[0], src[0], op.reg, 1]),
                }[op.op]
                emit(instruction, fields)
                continue
            elif op.op in ("sfpaddi", "sfpmuli"):
                fields = [op.offset, op.reg, 0]
            elif op.op == "sfpiadd":
                fields = [op.offset & 4095, src[0], op.reg, 5]
            elif op.op == "sfpsetexp":
                fields = [0, src[0], op.reg, 0]
            elif op.op == "sfpshft2":
                fields = [0, src[0], op.reg, 3]
            else:
                raise AssertionError(op.op)
            emit(op.op.upper(), fields)
        lines.append("}")
        if replay:
            if not 1 <= instruction_count <= 32:
                self.fail(fn, "replay bodies require 1..32 instructions")
            lines.append(f"static constexpr uint32_t {fn.name}_instruction_count = {instruction_count}u;")
        lines.insert(1, f"// Written LRegs (including outputs): {sorted(clobbers)}")
        return "\n".join(lines)

    def region(self, stmt):
        if len(stmt.items) != 1 or stmt.items[0].optional_vars:
            self.fail(stmt, "expected one predicate without an as binding")
        context = stmt.items[0].context_expr
        if not isinstance(context, ast.Call) or not isinstance(context.func, ast.Name) or context.func.id not in ("sfpiadd", "sfpsetcc"):
            self.fail(stmt, 'predicate must be sfpiadd or sfpsetcc with an explicit cc')
        before = self.env.copy()
        start = len(self.ops)
        self.call(context, predicate=True)
        for inner in stmt.body:
            if isinstance(inner, ast.Assign) and len(inner.targets) == 1 and isinstance(inner.targets[0], ast.Name):
                name = inner.targets[0].id
                if name in self.scalars or self.is_scalar(inner.value):
                    self.fail(inner, "scalar assignments are not allowed in SFPU predicates")
                if name == "ZERO":
                    self.fail(inner, "architectural constants cannot be rebound")
                self.env[name] = self.expr(inner.value)
            elif isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call):
                if isinstance(inner.value.func, ast.Name) and inner.value.func.id == "setrwc":
                    self.fail(inner, "setrwc is unconditional and cannot appear in a predicate")
                self.call(inner.value, store=True)
            else:
                self.fail(inner, "predicate body supports assignments and stores only; nesting is unsupported")
        copies = []
        merged = before.copy()
        for name, old in before.items():
            new = self.env[name]
            if new is not old:
                slot = Value("copy", [old], stmt.lineno)
                copies.append(slot)
                merge = Value("merge", [slot, new], stmt.lineno)
                self.ops.append(merge)
                merged[name] = merge
        # Seed inactive lanes before enabling predication. Region-local names do not escape.
        self.ops[start:start] = copies
        self.ops.append(Value("end", line=stmt.lineno))
        self.env = merged

    def expr(self, node):
        if isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id]
        if isinstance(node, ast.Call):
            return self.call(node)
        self.fail(node, "expected a defined value or SFPU call")

    def call(self, node, store=False, predicate=False):
        signatures = {"sfpload": ("offset",), "sfpstore": ("value", "offset"), "setrwc": (),
                      "sfpmin": ("a", "b"), "sfpmax": ("a", "b"), "sfpabs": ("a",),
                      "sfpneg": ("a",), "sfpsetsgn": ("a",), "sfprecip": ("a",), "sfpround": ("a",), "sfpsetcc": ("a",),
                      "sfpadd": ("a", "b"), "sfpmul": ("a", "b"), "sfpmad": ("a", "b", "c"),
                      "sfploadi": ("imm",), "sfpmuli": ("a", "imm"),
                      "sfpiadd": ("a", "imm"), "sfpsetexp": ("a", "exponent"), "sfpshft2": ("a",)}
        if not isinstance(node.func, ast.Name) or node.func.id not in signatures:
            self.fail(node, "unsupported SFPU instruction")
        name = node.func.id
        if (name in ("sfpstore", "setrwc")) != store:
            self.fail(node, "stores and setrwc must be standalone; arithmetic must produce a value")
        params = signatures[name]
        if len(node.args) > len(params):
            self.fail(node, "too many operands")
        values = dict(zip(params, node.args))
        for kw in node.keywords:
            options = {"sfpload": ("fmt", "addrmod"), "sfpstore": ("fmt", "addrmod"), "setrwc": ("dst",),
                       "sfpadd": ("negate_b",), "sfpmad": ("negate_product",), "sfpiadd": ("cc",),
                       "sfpsetcc": ("cc",), "sfpsetsgn": ("sign",), "sfpround": ("fmt", "rounding"), "sfpshft2": ("mode",)}.get(name, ())
            if kw.arg in values or kw.arg not in (*params, *options):
                self.fail(kw, "unknown or repeated operand")
            values[kw.arg] = kw.value
        if not all(p in values for p in params):
            self.fail(node, "missing operand")
        if name == "setrwc":
            dst = self.literal(values["dst"]) if "dst" in values else None
            if type(dst) is not int or not 0 <= dst <= 15:
                self.fail(node, "setrwc requires a literal dst in 0..15")
            self.ops.append(Value("setrwc", line=node.lineno, offset=dst))
            return
        memory = name in ("sfpload", "sfpstore")
        op = Value(name, line=node.lineno)
        # Evaluate operand expressions in written order, including named operands.
        evaluated = {}
        for key, value in values.items():
            if key not in ("offset", "fmt", "rounding", "imm", "sign", "negate_b", "negate_product", "cc", "mode", "addrmod"):
                evaluated[key] = self.expr(value)
        op.args = [evaluated[p] for p in params if p not in ("offset", "imm")]
        if name == "sfpsetsgn":
            sign = self.literal(values["sign"]) if "sign" in values else None
            if type(sign) is not int or sign not in (0, 1):
                self.fail(node, "sfpsetsgn requires a literal sign of 0 or 1")
            op.offset = sign
        if name == "sfpround":
            if ("fmt" not in values or self.literal(values["fmt"]) != "bf16" or
                    "rounding" not in values or self.literal(values["rounding"]) != "nearest"):
                self.fail(node, 'sfpround currently requires fmt="bf16", rounding="nearest"')
        if "imm" in values:
            imm = values["imm"]
            if name == "sfploadi" and self.is_scalar(imm):
                op.offset = self.scalar(imm)
            elif name == "sfploadi" and isinstance(imm, ast.Name) and imm.id in self.bits:
                op.offset = imm.id
            else:
                op.offset = self.literal(imm)
                lo, hi = (-2048, 2047) if name == "sfpiadd" else (0, 0xffffffff if name == "sfploadi" else 65535)
                if type(op.offset) is not int or not lo <= op.offset <= hi:
                    self.fail(node, f"immediate must be an integer in {lo}..{hi}")
        if name == "sfpshft2" and ("mode" not in values or self.literal(values["mode"]) != "rotate"):
            self.fail(node, 'sfpshft2 requires mode="rotate"')
        if name == "sfpadd":
            negate_b = self.literal(values["negate_b"]) if "negate_b" in values else False
            if type(negate_b) is not bool:
                self.fail(node, "negate_b must be a literal boolean")
            op.mod = 2 if negate_b else 0
        if name == "sfpmad":
            negate = self.literal(values["negate_product"]) if "negate_product" in values else False
            if type(negate) is not bool:
                self.fail(node, "negate_product must be a literal boolean")
            op.mod = int(negate)
        if name == "sfpsetcc":
            cc = self.literal(values["cc"]) if "cc" in values else None
            if not predicate or cc not in ('lt0', 'gte0', 'eq0', 'ne0'):
                self.fail(node, "sfpsetcc requires a with predicate and explicit cc")
            op.op = 'begin_cc'
            op.mod = {'lt0': 0, 'gte0': 4, 'eq0': 6, 'ne0': 2}[cc]
        if name == "sfpiadd":
            cc = self.literal(values["cc"]) if "cc" in values else "none"
            if (predicate and cc not in ("gte0", "lt0")) or (not predicate and cc != "none"):
                self.fail(node, "condition codes may only be set by a with predicate")
            if predicate:
                op.op = "begin"
                op.mod = 9 if cc == "gte0" else 1
        if memory:
            mod = values.get("addrmod")
            if isinstance(mod, ast.Name) and mod.id in self.addrmods:
                op.mod = mod.id
            else:
                op.mod = self.literal(mod) if mod is not None else 0
            if op.mod not in self.addrmods and (type(op.mod) is not int or not 0 <= op.mod <= 3):
                self.fail(node, "addrmod must be an integer in 0..3")
            op.offset = self.address(values["offset"])
            fmt = self.literal(values["fmt"]) if "fmt" in values else "fp32"
            if fmt not in ("fp32", "bf16"):
                self.fail(node, "fmt must be 'fp32' or 'bf16'")
            op.fmt = {"fp32": 3, "bf16": 2}[fmt]
        if name == "sfploadi" and type(op.offset) is int:
            for reg, bits in self.constant_values.items():
                if op.offset == bits:
                    return Value("input", reg=reg)
        self.ops.append(op)
        return op


EXAMPLES = '''@sfpu()
def residual_add():
    a = sfpload(0)
    b = sfpload(2)
    c = sfpload(4)
    d = sfpload(6)
    sfpstore(sfpadd(a, c), 0)
    sfpstore(sfpadd(b, d), 2)

@sfpu(inputs={"scale": 6}, preserve=(6,))
def scale_weight(scale):
    a = sfpload(0)
    b = sfpload(2)
    c = sfpload(4)
    d = sfpload(6)
    a = sfpmul(a, scale)
    b = sfpmul(b, scale)
    sfpstore(sfpmul(a, c), 0)
    sfpstore(sfpmul(b, d), 2)

@sfpu()
def rope():
    a = sfpload(0)
    b = sfpload(2)
    c = sfpload(4)
    d = sfpload(6)
    e = sfpload(8)
    f = sfpload(10)
    g = sfpload(12)
    h = sfpload(14)
    a = sfpmul(a, e)
    a = sfpmad(c, g, a)
    b = sfpmul(b, f)
    b = sfpmad(d, h, b)
    sfpstore(a, 0)
    sfpstore(b, 2)

@sfpu(inputs={"v0": 6, "v1": 7, "scale": 4}, preserve=(6, 7))
def weighted_value(v0, v1, scale):
    a = sfpload(0)
    b = sfpload(2)
    a = sfpmad(v0, scale, a)
    b = sfpmad(v1, scale, b)
    sfpstore(a, 0)
    sfpstore(b, 2)

@sfpu(inputs={"acc0": 4, "acc1": 5}, outputs={"out0": 4, "out1": 5})
def squares(acc0, acc1):
    a = sfpload(0)
    b = sfpload(2)
    out0 = sfpmad(a, a, acc0)
    out1 = sfpmad(b, b, acc1)
    return out0, out1
'''


class Tests(unittest.TestCase):
    def test_tuple_helpers(self):
        helper = '''@sfpu_helper()
def swap(x,y):
    return y,x
@sfpu_helper()
def twice(x,y):
    x,y=swap(x,y)
    return swap(x,y)
'''
        entry = "@sfpu(inputs={'x':0,'y':1})\ndef f(x,y):\n    x,y=swap(x,y)\n    sfpstore(x,0)\n    sfpstore(y,2)\n"
        code = Compiler().compile(helper + entry)
        self.assertIn('TTI_SFPSTORE(1u, 3u, 0u, 0u)', code)
        self.assertIn('TTI_SFPSTORE(0u, 3u, 0u, 2u)', code)
        code = Compiler().compile(helper + entry.replace('x,y=swap(x,y)', 'x,y=twice(x,y)'))
        self.assertIn('TTI_SFPSTORE(0u, 3u, 0u, 0u)', code)
        self.assertIn('TTI_SFPSTORE(1u, 3u, 0u, 2u)', code)
        for assignment in ('x=swap(x,y)', 'x,y,z=swap(x,y)', 'x,x=swap(x,y)',
                           'ZERO,y=swap(x,y)', 'x,y=(x,(x,y))'):
            with self.subTest(assignment=assignment), self.assertRaises(CompileError):
                Compiler().compile(helper + entry.replace('x,y=swap(x,y)', assignment))

    def test_paired_helper_codegen(self):
        production = Path(__file__).resolve().parents[1] / 'src/kernels.sfpu'
        code = Compiler().compile(production.read_text())
        body = code.split('void sfpu_reduce_subvec_sum_pair_body(void) {')[1].split('\n}')[0]
        self.assertEqual(body.count('TTI_SFPSHFT2('), 14)
        self.assertEqual(body.count('TTI_SFPADD('), 6)
        self.assertEqual(body.count('TTI_SFPNOP;'), 1)
        self.assertNotIn('TTI_SFPMOV', body)

    def test_helper_composition(self):
        helper = '''@sfpu_helper()
def square_plus(x, y):
    temporary = sfpmul(x, x)
    return sfpadd(temporary, y)
@sfpu_helper()
def outer(x, y):
    return square_plus(y=y, x=x)
'''
        entry = '''@sfpu(inputs={'a':0,'b':1})
def f(a,b):
    temporary = sfpmul(b,b)
    value = outer(sfpadd(a,b), temporary)
    sfpstore(temporary,0)
    sfpstore(value,2)
'''
        inline = entry.replace('    value = outer(sfpadd(a,b), temporary)',
                               '    x = sfpadd(a,b)\n    t = sfpmul(x,x)\n    value = sfpadd(t,temporary)')
        instructions = lambda code: re.findall(r'TTI_\w+\([^;]+;', code)
        self.assertEqual(instructions(Compiler().compile(helper + entry)),
                         instructions(Compiler().compile(inline)))
        self.assertNotIn('void square_plus', Compiler().compile(helper + entry))
        for call in ('outer(a)', 'outer(a,b,c=a)', 'outer(a,x=b)', 'outer(*a)'):
            with self.subTest(call=call), self.assertRaises(CompileError):
                Compiler().compile(helper + entry.replace('outer(sfpadd(a,b), temporary)', call))
        with self.assertRaises(CompileError):
            Compiler().compile(helper.replace('return square_plus(y=y, x=x)', 'return outer(x,y)') + entry)
        with self.assertRaises(CompileError):
            Compiler().compile(helper.replace('return square_plus(y=y, x=x)', 'return undefined') + entry)

    def test_helper_mask_and_evaluation_order(self):
        helper = '''@sfpu_helper()
def mask(x, test):
    result = x
    with sfpiadd(test,0,cc='gte0'):
        result = sfpmul(x,x)
    return result
'''
        entry = "@sfpu()\ndef f():\n    r=mask(test=sfpload(2),x=sfpload(0))\n    sfpstore(r,4)\n"
        inline = ("@sfpu()\ndef f():\n    test=sfpload(2)\n    x=sfpload(0)\n"
                  "    result=x\n    with sfpiadd(test,0,cc='gte0'):\n        result=sfpmul(x,x)\n"
                  "    sfpstore(result,4)\n")
        instructions = lambda code: re.findall(r'TTI_\w+\([^;]+;', code)
        self.assertEqual(instructions(Compiler().compile(helper+entry)), instructions(Compiler().compile(inline)))
        composed = ("@sfpu_helper()\ndef identity(x):\n    return x\n"
                    "@sfpu()\ndef f():\n    x=sfpadd(sfpload(0),identity(sfpload(2)))\n    sfpstore(x,4)")
        plain = "@sfpu()\ndef f():\n    x=sfpadd(sfpload(0),sfpload(2))\n    sfpstore(x,4)"
        self.assertEqual(instructions(Compiler().compile(composed)), instructions(Compiler().compile(plain)))

    def test_transpose_mapping(self):
        for sources in ((0, 1, 2, 3), (1, 0, 3, 2), (4, 5, 6, 7),
                        (0, 0, 0, 0), tuple(range(8))):
            count = len(sources)
            inputs = {f'a{i}': i for i in set(sources)}
            names = [f'b{i}' for i in range(count)]
            source = (f'@sfpu(inputs={inputs!r})\ndef f({",".join(inputs)}):\n    '
                      + ','.join(names) + '=sfptransp('
                      + ','.join(f'a{i}' for i in sources) + ')\n'
                      + ''.join(f'    sfpstore(b{i},{i})\n' for i in range(count)))
            code = Compiler().compile(source)
            before = [[r * 100 + lane for lane in range(32)] for r in range(8)]
            regs = [v[:] for v in before]
            stored = {}
            for op, fields in re.findall(r'TTI_(\w+)\(([^)]*)\)', code):
                a = [int(v.strip().removesuffix('u')) for v in fields.split(',')]
                if op == 'SFPMOV':
                    regs[a[2]] = regs[a[1]][:]
                elif op == 'SFPTRANSP':
                    old = [v[:] for v in regs]
                    for base in (0, 4):
                        for r in range(4):
                            regs[base+r] = [old[base+lane//8][r*8+lane%8] for lane in range(32)]
                else:
                    self.assertEqual(op, 'SFPSTORE')
                    stored[a[3]] = regs[a[0]][:]
            for r in range(count):
                expected = [before[sources[(r//4)*4+lane//8]][(r%4)*8+lane%8]
                            for lane in range(32)]
                self.assertEqual(stored[r], expected, sources)

    def test_transpose_rejections(self):
        prefix = "@sfpu(inputs={'a':0,'b':1,'c':2,'d':3})\ndef f(a,b,c,d):\n"
        for body in ('    w,x,y,z=sfptransp(a,b,c,d)\n    sfpstore(a,0)',
                     '    w,x,y=sfptransp(a,b,c)',
                     '    w,w,y,z=sfptransp(a,b,c,d)',
                     '    x=sfptransp(a,b,c,d)',
                     '    w,x,y,z=sfptransp(a,b,c,d,mode=0)',
                     '    with sfpiadd(a,0,cc="gte0"):\n        w,x,y,z=sfptransp(a,b,c,d)'):
            with self.subTest(body=body), self.assertRaises(CompileError):
                Compiler().compile(prefix + body)
        with self.assertRaises(CompileError):
            Compiler().compile(prefix.replace("})", "},preserve=(7,))") +
                               '    w,x,y,z=sfptransp(a,b,c,d)')

    def test_template_offsets(self):
        source = '''@sfpu(templates=('offset',))
def f():
    x = sfpload(offset * 4)
    sfpstore(x, offset * 4 + 2)
'''
        code = Compiler().compile(source)
        self.assertIn('template<uint32_t offset>', code)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.cpp'
            for value, valid in ((0, True), (2047, True), (2048, False), (8192, False)):
                path.write_text('#include <cstdint>\n'
                                '#define TTI_SFPLOAD(r,f,m,o) static_assert((o)>=0 && (o)<8192)\n'
                                '#define TTI_SFPSTORE(r,f,m,o) static_assert((o)>=0 && (o)<8192)\n'
                                + code + f'\nvoid test() {{ f<{value}>(); }}\n')
                result = subprocess.run(['g++', '-std=c++20', '-fsyntax-only', str(path)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, valid, result.stderr)
        for expression in ('unknown', 'offset * offset', 'offset / 2', 'True', 'sfpload(0)'):
            with self.subTest(expression=expression), self.assertRaises(CompileError):
                Compiler().compile(source.replace('offset * 4 + 2', expression))
        for body in ('offset = ZERO', 'x = sfpmul(offset, ZERO)'):
            with self.subTest(body=body), self.assertRaises(CompileError):
                Compiler().compile("@sfpu(templates=('offset',))\ndef f():\n    " + body)
        for names in ("'offset'", "('offset','offset')", "('ZERO',)", "('bad-name',)", "(1,)"):
            with self.subTest(names=names), self.assertRaises(CompileError):
                Compiler().compile(f'@sfpu(templates={names})\ndef f():\n    sfpstore(ZERO,0)')
        with self.assertRaises(CompileError):
            Compiler().compile("@sfpu(templates=('x',),inputs={'x':0})\ndef f(x):\n    sfpstore(x,0)")

    def test_runtime_offsets(self):
        source = '''@sfpu(templates=('base',), offsets=('row',), addrmods=('step',))
def f(row):
    x=sfpload(base,addrmod=step)
    sfpstore(x,row+base+2,addrmod=step)
    y=sfpload(row*4)
    sfpstore(y,4)
    sfpstore(x,row-row+6)
'''
        code = Compiler().compile(source)
        self.assertIn('void f(uint32_t row)', code)
        self.assertEqual(code.count('tensix_instruction(TT_OP_'), 2)
        self.assertEqual(code.count('TTI_SFP'), 3)
        self.assertNotIn('static_assert(row', code)
        header = Path(__file__).resolve().parents[1] / 'src/ckernel_ops.h'
        harness = f'''#include <cstdint>
#include <cassert>
#include "{header}"
static uint32_t words[5], count, dynamic_count;
static void immediate(uint32_t word) {{ words[count++]=word; }}
static void tensix_instruction(uint32_t word) {{ ++dynamic_count; immediate(word); }}
#undef INSTRUCTION_WORD
#define INSTRUCTION_WORD(x) immediate(x)
''' + code + '''
int main() {
    const uint32_t rows[] = {0,1,127,2047};
    for (uint32_t row : rows) {
        count=dynamic_count=0;
        f<16,3>(row);
        assert(count==5 && dynamic_count==2);
        assert(words[0]==TT_OP_SFPLOAD(0,3,3,16));
        assert(words[1]==TT_OP_SFPSTORE(0,3,3,row+18));
        assert(words[2]==TT_OP_SFPLOAD(1,3,0,row*4));
        assert(words[3]==TT_OP_SFPSTORE(1,3,0,4));
        assert(words[4]==TT_OP_SFPSTORE(0,3,0,6));
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.cpp'
            exe = Path(directory) / 'test'
            path.write_text(harness)
            result = subprocess.run(['g++', '-std=gnu++20', '-O2', '-Wall', '-Wextra', '-Werror',
                                     str(path), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        for body in ('row=ZERO', 'x=sfpmul(row,ZERO)', 'x=sfpload(row*row)',
                     'x=sfpload(row/2)', 'x=sfpload(unknown)', 'x=sfploadi(row)',
                     'setrwc(dst=row)', 'x=sfpload(0,addrmod=row)'):
            with self.subTest(body=body), self.assertRaises(CompileError):
                Compiler().compile("@sfpu(offsets=('row',))\ndef f(row):\n    " + body)
        for contract, args in (("offsets=('row','row')", 'row'),
                               ("offsets=('row',),templates=('row',)", 'row'),
                               ("offsets=('row',),inputs={'row':0}", 'row'),
                               ("offsets=('row',),outputs={'row':0}", 'row'),
                               ("offsets=('row',)", ''), ("offsets='row'", 'row'),
                               ("offsets=('ZERO',)", 'ZERO')):
            with self.subTest(contract=contract), self.assertRaises(CompileError):
                Compiler().compile(f'@sfpu({contract})\ndef f({args}):\n    sfpstore(ZERO,0)')

    def test_rotate_hazards(self):
        for expr in ('sfpadd(a,b)', 'sfpmul(a,b)', 'sfpmad(a,b,a)', 'sfpadd(a,sfploadi(0x3f800000))', 'sfpmuli(a,0x3f80)'):
            source = ('@sfpu(inputs={"a":0,"b":1})\ndef f(a,b):\n'
                      f'    x={expr}\n    y=sfpshft2(x,mode="rotate")\n    sfpstore(y,0)')
            with self.subTest(expr=expr):
                code = Compiler().compile(source)
                self.assertEqual(code.count('TTI_SFPNOP;'), 1)
                self.assertRegex(code, r'TTI_SFPNOP;[^\n]*\n    TTI_SFPSHFT2')
        for body, nops in (
            ('x=sfpshft2(a,mode="rotate")\n    x=sfpshft2(x,mode="rotate")', 1),
            ('x=sfpload(0)\n    x=sfpshft2(x,mode="rotate")', 0),
            ('x=sfpadd(a,b)\n    sfpstore(b,0)\n    x=sfpshft2(x,mode="rotate")', 0),
            ('x=sfpadd(a,b)\n    x=sfpshft2(b,mode="rotate")', 0),
        ):
            with self.subTest(body=body):
                code = Compiler().compile('@sfpu(inputs={"a":0,"b":1})\ndef f(a,b):\n    '
                                          + body + '\n    sfpstore(x,0)')
                self.assertEqual(code.count('TTI_SFPNOP;'), nops)

    def test_add_immediate_fusion(self):
        for bits in (0, 0x80000000, 0x3f800000, 0x40000000, 0xbf800000):
            for expression in (f'sfpadd(a,sfploadi({bits}))', f'sfpadd(sfploadi({bits}),a)'):
                code = Compiler().compile('@sfpu(inputs={"a":3},outputs={"b":3})\ndef f(a):\n'
                                          f'    b={expression}\n    return b')
                self.assertIn(f'TTI_SFPADDI({bits >> 16}u, 3u, 0u)', code)
                self.assertEqual(code.count('TTI_'), 1)
        for expression in ('sfpadd(a,sfploadi(0x3f800001))', 'sfpadd(a,sfploadi(0x7f800000))',
                           'sfpadd(a,sfploadi(0x7fc10000))', 'sfpadd(a,sfploadi(0x3f800000),negate_b=True)'):
            code = Compiler().compile('@sfpu(inputs={"a":0})\ndef f(a):\n'
                                      f'    b={expression}\n    sfpstore(b,0)')
            self.assertIn('TTI_SFPADD(', code)
        code = Compiler().compile('@sfpu(inputs={"a":0})\ndef f(a):\n'
                                  '    b=sfpadd(a,sfploadi(0x3f800000))\n    sfpstore(a,0)\n    sfpstore(b,2)')
        self.assertNotIn('TTI_SFPADDI', code)
        code = Compiler().compile('@sfpu(inputs={"a":0},outputs={"b":4})\ndef f(a):\n'
                                  '    b=sfpadd(a,sfploadi(0x3f800000))\n    return b')
        self.assertNotIn('TTI_SFPADDI', code)
        code = Compiler().compile('@sfpu(inputs={"a":0})\ndef f(a):\n'
                                  '    k=sfploadi(0x40000000)\n    b=sfpadd(a,k)\n'
                                  '    c=sfpshft2(b,mode="rotate")\n    sfpstore(k,0)\n    sfpstore(c,2)')
        self.assertIn('TTI_SFPLOADI', code)
        self.assertIn('TTI_SFPADDI', code)
        self.assertIn('TTI_SFPNOP;', code)
        with self.assertRaises(CompileError):
            Compiler().compile('@sfpu()\ndef f():\n    sfpstore(FP32_ONE,0)')
        with self.assertRaises(CompileError):
            Compiler().compile('@sfpu(inputs={"a":0})\ndef f(a):\n    b=sfpaddi(a,0x3f80)\n    sfpstore(b,0)')
        code = Compiler().compile('@sfpu(inputs={"a":0})\ndef f(a):\n'
                                  '    b=sfpadd(a,sfploadi(0x41a00000))\n    sfpstore(a,0)\n    sfpstore(b,2)')
        self.assertNotIn('TTI_SFPLOADI', code)
        self.assertIn('TTI_SFPMOV(0u, 0u, 1u, 0u)', code)
        self.assertIn('TTI_SFPADDI(16800u, 1u, 0u)', code)

    def test_multiply_immediate(self):
        for imm in (0, 0x8000, 0x3e00, 0x7f80, 0x7fc1, 0xffff):
            code = Compiler().compile('@sfpu(inputs={"a":4},outputs={"b":4})\ndef f(a):\n'
                                      f'    b=sfpmuli(a,imm={imm})\n    return b')
            self.assertIn(f'TTI_SFPMULI({imm}u, 4u, 0u)', code)
            self.assertNotIn('TTI_SFPMOV', code)
        code = Compiler().compile('@sfpu(inputs={"a":4})\ndef f(a):\n'
                                  '    b=sfpmuli(a,0x3e00)\n    sfpstore(a,0)\n    sfpstore(b,2)')
        self.assertIn('TTI_SFPMOV(0u, 4u, 0u, 0u)', code)
        self.assertIn('TTI_SFPMULI(15872u, 0u, 0u)', code)
        for imm in ('-1', '65536', 'True', '1.0', 'a'):
            with self.subTest(imm=imm), self.assertRaises(CompileError):
                Compiler().compile('@sfpu(inputs={"a":0})\ndef f(a):\n'
                                   f'    b=sfpmuli(a,{imm})\n    sfpstore(b,0)')
        code = Compiler().compile('@sfpu(inputs={"a":0})\ndef f(a):\n'
                                  '    b=sfpmul(a,sfploadi(0x3e000000))\n    sfpstore(b,0)')
        self.assertNotIn('TTI_SFPMULI', code)
        self.assertIn('TTI_SFPMUL(', code)

    def test_replay(self):
        code = Compiler().compile('''@sfpu(replay=True, templates=('row',), inputs={'x':0}, outputs={'y':4}, preserve=(0,))
def body(x):
    y=sfpshft2(x,mode='rotate')
    sfpstore(y,row)
    setrwc(dst=0)
    return y
''')
        self.assertIn('TTI_SFPNOP;', code)
        count = len(re.findall(r'^    TTI_', code, re.M))
        self.assertIn(f'body_instruction_count = {count}u;', code)
        self.assertNotIn('tensix_instruction(', code)
        code = Compiler().compile('@sfpu(replay=True)\ndef f():\n    sfpstore(sfploadi(0xdeadbeef),0)')
        self.assertIn('f_instruction_count = 3u;', code)
        code = Compiler().compile("@sfpu(replay=True,inputs={'x':0},outputs={'y':1},preserve=(0,))\n"
                                  "def f(x):\n    y=x\n    return y")
        self.assertIn('TTI_SFPMOV', code)
        self.assertIn('f_instruction_count = 1u;', code)
        code = Compiler().compile('@sfpu(replay=True)\ndef f():\n    '
                                  + '\n    '.join(['sfpstore(ZERO,0)'] * 32))
        self.assertIn('f_instruction_count = 32u;', code)
        for contract, args, body in (
                ('replay=1', '', 'sfpstore(ZERO,0)'),
                ("replay=True,bits=('x',)", 'x', 'sfpstore(sfploadi(x),0)'),
                ("replay=True,offsets=('x',)", 'x', 'sfpstore(ZERO,x)'),
                ('replay=True', '', 'x=add32(1,2)\n    sfpstore(ZERO,0)'),
                ('replay=True', '', 'return'),
                ('replay=True', '', '\n    '.join(['sfpstore(ZERO,0)'] * 33)),
                ('replay=True', '', 'replay(0,1)')):
            with self.subTest(contract=contract, body=body), self.assertRaises(CompileError):
                Compiler().compile(f'@sfpu({contract})\ndef f({args}):\n    {body}')

    def test_runtime_bits(self):
        code = Compiler().compile("@sfpu(bits=('pattern',),outputs={'x':3})\ndef f(pattern):\n"
                                  "    x=sfploadi(pattern)\n    return x")
        self.assertEqual(code.count('tensix_instruction(TT_OP_SFPLOADI'), 1)
        self.assertIn('pack_u16(pattern, TT_OP_SFPLOADI(3u, 10u, 0u) >> 16)', code)
        header = Path(__file__).resolve().parents[1] / 'src/ckernel_ops.h'
        harness = f'''#include <cstdint>
#include <cassert>
#include "{header}"
static uint32_t words[2], count;
static uint32_t pack_u16(uint32_t low, uint32_t high) {{ return (low & 65535u) | (high << 16); }}
static void tensix_instruction(uint32_t word) {{ words[count++]=word; }}
''' + code + '''
int main() {
    for (uint32_t value : {0u, 1u, 0x80000000u, 0x3f800000u, 0x7f800000u, 0x7fc12345u, 0xffffffffu}) {
        count=0;
        f(value);
        assert(count==2);
        assert(words[0]==TT_OP_SFPLOADI(3,8,value>>16));
        assert(words[1]==TT_OP_SFPLOADI(3,10,value&65535u));
    }
}
'''
        harness = '#include <initializer_list>\n' + harness
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.cpp'
            exe = Path(directory) / 'test'
            path.write_text(harness)
            result = subprocess.run(['g++', '-std=gnu++20', '-O2', '-Wall', '-Wextra', '-Werror',
                                     str(path), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        for body in ('x=sfploadi(pattern+1)', 'x=sfpload(pattern)', 'x=sfpmul(pattern,sfploadi(0x3f800000))',
                     'x=sfpmuli(sfploadi(0x3f800000),pattern)', 'pattern=sfpload(0)', 'setrwc(dst=pattern)'):
            with self.subTest(body=body), self.assertRaises(CompileError):
                Compiler().compile("@sfpu(bits=('pattern',))\ndef f(pattern):\n    " + body)
        for contract in ("bits='pattern'", "bits=('pattern','pattern')", "bits=('ZERO',)",
                         "bits=('pattern',),offsets=('pattern',)", "bits=('pattern',),templates=('pattern',)"):
            with self.subTest(contract=contract), self.assertRaises(CompileError):
                Compiler().compile("@sfpu(" + contract + ")\ndef f(pattern):\n    x=sfploadi(pattern)")

    def test_scalar_arithmetic(self):
        code = Compiler().compile("""@sfpu(bits=('a','b'))
def f(a,b):
    total = add32(a,b)
    saved = total
    total = sub32(total,1)
    x = sfploadi(saved)
    sfpstore(x,0)
    y = sfploadi(total)
    sfpstore(y,2)

@sfpu(offsets=('maximum','row'))
def address(maximum,row):
    x = sfpload(sub32(add32(maximum,2),row))
    sfpstore(x,0)
""")
        header = Path(__file__).resolve().parents[1] / 'src/ckernel_ops.h'
        harness = f"""#include <cstdint>
#include <cassert>
#include "{header}"
static uint32_t words[6], count;
static uint32_t pack_u16(uint32_t low, uint32_t high) {{ return (low & 65535u) | (high << 16); }}
static void tensix_instruction(uint32_t word) {{ words[count++]=word; }}
#undef INSTRUCTION_WORD
#define INSTRUCTION_WORD(x) tensix_instruction(x)
""" + code + """
int main() {
    const uint32_t values[] = {0,1,0x7fffffff,0x80000000,0xffffffff};
    for (uint32_t a : values) {
        for (uint32_t b : values) {
            count=0;
            f(a,b);
            assert(count==6);
            uint32_t sum=a+b, difference=sum-1;
            assert(words[0]==TT_OP_SFPLOADI(0,8,sum>>16));
            assert(words[1]==TT_OP_SFPLOADI(0,10,sum&65535));
            assert(words[3]==TT_OP_SFPLOADI(0,8,difference>>16));
            assert(words[4]==TT_OP_SFPLOADI(0,10,difference&65535));
        }
    }
    for (uint32_t row=0; row<256; row+=4) {
        count=0;
        address(256,row);
        assert(count==2);
        assert(words[0]==TT_OP_SFPLOAD(0,3,0,258-row));
    }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.cpp'
            exe = Path(directory) / 'test'
            path.write_text(harness)
            result = subprocess.run(['g++', '-std=gnu++20', '-O2', '-Wall', '-Wextra', '-Werror',
                                     str(path), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        for body in ('s=add32(a)', 's=add32(a,True)', 's=sub32(a,-1)', 's=add32(a,0x100000000)',
                     's=add32(a=1,b=2)', 's=add32(x,1)', 's=a+1',
                     'x=add32(a,1)', 's=add32(a,1)\n    s=sfpload(0)',
                     's=add32(a,1)\n    y=sfpmul(s,x)',
                     'with sfpsetcc(x,cc="lt0"):\n        s=add32(a,1)'):
            with self.subTest(body=body), self.assertRaises(CompileError):
                Compiler().compile('@sfpu(bits=("a",),inputs={"x":0})\ndef f(a,x):\n    '+body)

    def test_constant_contract(self):
        source = """@sfpu_helper()
def coefficient():
    return sfploadi(0x3e2aaaab)

@sfpu(constants={11:0x3e2aaaab})
def f():
    x=coefficient()
    sfpstore(x,0)

@sfpu()
def g():
    x=coefficient()
    sfpstore(x,0)
"""
        code = Compiler().compile(source)
        first, second = code.split('static inline void g')
        self.assertIn('LReg11=0x3e2aaaab', first)
        self.assertIn('TTI_SFPSTORE(11u, 3u, 0u, 0u)', first)
        self.assertNotIn('TTI_SFPLOADI', first)
        self.assertEqual(second.count('TTI_SFPLOADI'), 2)
        code = Compiler().compile('@sfpu(constants={12:0x3e2aaaab})\ndef f():\n'
                                  '    x=sfpmuli(sfploadi(0x3e2aaaab),0x4000)\n    sfpstore(x,0)')
        self.assertIn('TTI_SFPMOV(0u, 12u, 0u, 0u)', code)
        self.assertNotIn('TTI_SFPCONFIG', code)
        for contract in ('constants={10:0}', 'constants={15:0}', 'constants={11:-1}',
                         'constants={11:True}', 'constants={11:0x100000000}', 'constants=[]',
                         'constants={11:0,11:1}', 'constants={11:0},inputs={"x":11}'):
            with self.subTest(contract=contract), self.assertRaises(CompileError):
                Compiler().compile('@sfpu('+contract+')\ndef f():\n    sfpstore(ZERO,0)')

    def test_constant_outputs(self):
        code = Compiler().compile('@sfpu(outputs={"a":4,"b":5})\ndef f():\n'
                                  '    a=ZERO\n    b=ZERO\n    return a,b')
        self.assertEqual(code.count('TTI_'), 2)
        self.assertIn('TTI_SFPMOV(0u, 9u, 4u, 2u)', code)
        self.assertIn('TTI_SFPMOV(0u, 9u, 5u, 2u)', code)
        code = Compiler().compile('@sfpu(inputs={"x":2}, outputs={"y":4}, preserve=(2,))\n'
                                  'def f(x):\n    y=x\n    return y')
        self.assertEqual(code.count('TTI_'), 1)
        self.assertIn('TTI_SFPMOV(0u, 2u, 4u, 2u)', code)

    def test_subvector_reduction(self):
        source = Path(__file__).resolve().parents[1] / 'src/kernels.sfpu'
        fn = next(n for n in ast.parse(source.read_text()).body if n.name == 'reduce_subvec_sum')
        code = Compiler().compile(ast.unparse(fn) + '''
@sfpu(inputs={'x':4},outputs={'x':4},preserve=(1,2,3,5,6,7))
def test_reduce(x):
    x=reduce_subvec_sum(x)
    return x
''')
        self.assertEqual(code.count('TTI_SFPNOP;'), 3)
        self.assertEqual(code.count('TTI_SFPSHFT2('), 7)
        self.assertEqual(code.count('TTI_SFPADD('), 3)
        self.assertNotIn('TTI_SFPMOV(', code)
        self.assertIn('Written LRegs (including outputs): [0, 4]', code)
        regs = {r: [100*r + lane for lane in range(32)] for r in range(8)}
        before = {r: v.copy() for r, v in regs.items()}
        expected = [sum(before[4][lane & ~7:(lane & ~7) + 8]) for lane in range(32)]
        for op, operands in re.findall(r'TTI_(\w+)\(([^)]*)\)', code):
            a = [int(x.strip().removesuffix('u')) for x in operands.split(',')]
            if op == 'SFPSHFT2':
                values = regs[a[1]]
                regs[a[2]] = [values[(lane & ~7) + ((lane - 1) & 7)] for lane in range(32)]
            else:
                self.assertEqual(op, 'SFPADD')
                regs[a[3]] = [x + y for x, y in zip(regs[a[1]], regs[a[2]])]
        self.assertEqual(regs[4], expected)
        for r in (1, 2, 3, 5, 6, 7):
            self.assertEqual(regs[r], before[r])

    def test_squares_reduction(self):
        source = Path(__file__).resolve().parents[1] / 'src/kernels.sfpu'
        nodes = [n for n in ast.parse(source.read_text()).body
                 if n.name in ('reduce_subvec_sum', 'sfpu_squares_reduce_body')]
        code = Compiler().compile('\n'.join(ast.unparse(n) for n in nodes))
        self.assertEqual(re.findall(r'TTI_(\w+)\(', code)[0], 'SETRWC')
        self.assertEqual(code.count('TTI_SFPADD('), 7)
        self.assertEqual(code.count('TTI_SFPMOV('), 3)
        self.assertEqual(code.count('TTI_SFPSHFT2('), 7)
        self.assertEqual(code.count('TTI_SFPNOP;'), 3)
        # Symbolic adds verify operand order and grouping, not just the total sum.
        regs = [[(r, lane) for lane in range(32)] for r in range(8)]
        def add(a, b):
            return ('+', a, b)
        partials = [add(regs[5][lane], regs[4][lane]) for lane in range(32)]
        expected = [add(partials[24+j], add(partials[16+j], add(partials[8+j], partials[j])))
                    for j in range(8)] * 4
        for shift in (1, 2, 4):
            expected = [add(expected[(lane & ~7) + ((lane-shift) & 7)], expected[lane])
                        for lane in range(32)]
        stored = {}
        for op, fields in re.findall(r'TTI_(\w+)\(([^)]*)\)', code):
            a = [int(v.strip().removesuffix('u')) for v in fields.split(',')]
            if op == 'SETRWC':
                self.assertEqual(a, [0, 0, 0, 0, 0, 4])
            elif op == 'SFPMOV':
                regs[a[2]] = regs[a[1]][:]
            elif op == 'SFPTRANSP':
                old = [v[:] for v in regs]
                for base in (0, 4):
                    for r in range(4):
                        regs[base+r] = [old[base+lane//8][r*8+lane%8] for lane in range(32)]
            elif op == 'SFPADD':
                regs[a[3]] = [add(x, y) for x, y in zip(regs[a[1]], regs[a[2]])]
            elif op == 'SFPSHFT2':
                values = regs[a[1]]
                regs[a[2]] = [values[(lane & ~7) + ((lane-1) & 7)] for lane in range(32)]
            else:
                self.assertEqual(op, 'SFPSTORE')
                stored[a[3]] = regs[a[0]][:]
        self.assertEqual(stored, {0: expected})

    def test_setrwc(self):
        for dst in (0, 1, 15):
            code = Compiler().compile(f'''@sfpu(inputs={{'x':3}},outputs={{'x':3}},preserve=(0,1,2,4,5,6,7))
def f(x):
    sfpstore(x,2)
    setrwc(dst={dst})
    sfpstore(x,0)
    return x
''')
            self.assertEqual(re.findall(r'TTI_(\w+)\(', code), ['SFPSTORE', 'SETRWC', 'SFPSTORE'])
            self.assertIn(f'TTI_SETRWC(0u, 0u, {dst}u, 0u, 0u, 4u)', code)
            self.assertIn('Written LRegs (including outputs): []', code)
        helper = '''@sfpu_helper()
def reset(x):
    setrwc(dst=0)
    return x
'''
        code = Compiler().compile(helper + '''@sfpu(inputs={'x':0})
def f(x):
    x=sfpmul(x,x)
    x=reset(x)
    x=sfpshft2(x,mode='rotate')
    sfpstore(x,0)
''')
        self.assertRegex(code, r'TTI_SETRWC[^\n]*\n    TTI_SFPNOP;[^\n]*\n    TTI_SFPSHFT2')
        for statement in ('setrwc()', 'setrwc(0)', 'setrwc(dst=-1)', 'setrwc(dst=16)',
                          'setrwc(dst=True)', 'setrwc(dst=x)', 'setrwc(dst=0,srca=0)',
                          'x=setrwc(dst=0)', 'x=sfpmul(x,setrwc(dst=0))',
                          "with sfpsetcc(x,cc='lt0'):\n        setrwc(dst=0)",
                          "with sfpsetcc(x,cc='lt0'):\n        x=reset(x)"):
            with self.subTest(statement=statement), self.assertRaises(CompileError):
                Compiler().compile(helper + "@sfpu(inputs={'x':0})\ndef f(x):\n    " + statement)

    def test_swiglu_operations(self):
        for op, instruction in (("sfpabs", "SFPABS"), ("sfpneg", "SFPMOV"),
                                ("sfprecip", "SFPARECIP")):
            code = Compiler().compile('@sfpu(inputs={"x":3},outputs={"y":5})\ndef f(x):\n'
                                      f'    y={op}(x)\n    return y')
            modifier = 0 if op == "sfprecip" else 1
            self.assertIn(f'TTI_{instruction}(0u, 3u, 5u, {modifier}u)', code)
        for op, modifier in (("sfpmin", 1), ("sfpmax", 9)):
            code = Compiler().compile('@sfpu(inputs={"x":0,"bound":11})\ndef f(x,bound):\n'
                                      f'    sfpstore({op}(sfpmul(x,x),bound),0)')
            self.assertRegex(code, rf'TTI_SFPNOP;[^\n]*\n    TTI_SFPSWAP\(0u, 11u, 0u, {modifier}u\)')
            with self.assertRaises(CompileError):
                Compiler().compile('@sfpu(inputs={"x":0,"y":1})\ndef f(x,y):\n'
                                   f'    sfpstore({op}(x,y),0)')
        for cc, modifier in (("lt0", 0), ("gte0", 4), ("eq0", 6), ("ne0", 2)):
            code = Compiler().compile('@sfpu(inputs={"x":3})\ndef f(x):\n'
                                      f'    with sfpsetcc(x,cc="{cc}"):\n        sfpstore(x,0)')
            self.assertIn(f'TTI_SFPSETCC(0u, 3u, 0u, {modifier}u)', code)

    def test_set_sign(self):
        for sign in (0, 1):
            code = Compiler().compile('@sfpu(inputs={"x":3},outputs={"y":5})\ndef f(x):\n'
                                      f'    y=sfpsetsgn(x,sign={sign})\n    return y')
            self.assertIn(f'TTI_SFPSETSGN({sign}u, 3u, 5u, 1u)', code)
            self.assertNotIn('TTI_SFPMOV', code)
        for options in ('', 'sign=-1', 'sign=2', 'sign=True', 'sign=x', 'sign="negative"'):
            with self.subTest(options=options), self.assertRaises(CompileError):
                Compiler().compile('@sfpu(inputs={"x":0})\ndef f(x):\n'
                                   f'    y=sfpsetsgn(x,{options})')

    def test_round(self):
        code = Compiler().compile('@sfpu(inputs={"x":0})\ndef f(x):\n'
                                  '    sfpstore(sfpround(x,fmt="bf16",rounding="nearest"),0)')
        self.assertIn('TTI_SFP_STOCH_RND(0u, 0u, 0u, 0u, 0u, 1u)', code)
        code = Compiler().compile('@sfpu(inputs={"x":3},outputs={"y":5})\ndef f(x):\n'
                                  '    y=sfpround(x,fmt="bf16",rounding="nearest")\n    return y')
        self.assertIn('TTI_SFP_STOCH_RND(0u, 0u, 3u, 3u, 5u, 1u)', code)
        for options in ('', 'fmt="bf16"', 'fmt="fp32",rounding="nearest"',
                        'fmt="bf16",rounding="stochastic"'):
            with self.subTest(options=options), self.assertRaises(CompileError):
                Compiler().compile('@sfpu(inputs={"x":0})\ndef f(x):\n'
                                   f'    y=sfpround(x,{options})')

    def test_negate_b(self):
        for flag, modifier in (("True", 2), ("False", 0)):
            code = Compiler().compile('@sfpu(inputs={"a":0,"b":1})\ndef f(a,b):\n'
                                      f'    sfpstore(sfpadd(a,b,negate_b={flag}),0)')
            self.assertIn(f"TTI_SFPADD(10u, 0u, 1u, 0u, {modifier}u)", code)
        for option in ('negate_b=1', 'negate_b="True"', 'mode="sub"'):
            with self.subTest(option=option), self.assertRaises(CompileError):
                Compiler().compile('@sfpu()\ndef f():\n'
                                   f'    x=sfpadd(ZERO,sfploadi(0x3f800000),{option})')

    def test_masked_live_old_value(self):
        source = '''@sfpu(inputs={"x":0,"test":1}, outputs={"x":7})
def f(x,test):
    old=x
    with sfpiadd(test,0,cc="gte0"):
        x=sfpmul(x,x)
    sfpstore(old,0)
    sfpstore(x,2)
    return x
'''
        code = Compiler().compile(source)
        instructions = re.findall(r"TTI_(\w+)\(([^)]*)\)", code)
        for test in (-1, 0, 1):
            regs = {i: 100+i for i in range(15)}
            regs.update({0: 3, 1: test, 9: 0, 10: 1})
            active = True
            dst = {}
            for op, operands in instructions:
                a = [int(x.strip().removesuffix("u")) for x in operands.split(",")]
                if op == "SFPENCC":
                    active = True
                elif op == "SFPMOV":
                    if active or a[3] == 2:
                        regs[a[2]] = regs[a[1]]
                elif op == "SFPIADD":
                    imm = a[0] if a[0] < 2048 else a[0]-4096
                    regs[a[2]] = regs[a[1]] + imm
                    active = regs[a[2]] >= 0
                elif op == "SFPMUL":
                    if active:
                        regs[a[3]] = regs[a[0]] * regs[a[1]]
                elif op == "SFPSTORE":
                    if active:
                        dst[a[3]] = regs[a[0]]
                else:
                    self.fail(op)
            self.assertEqual(dst, {0: 3, 2: 9 if test >= 0 else 3})
            self.assertEqual(regs[7], dst[2])

    def test_masked_inplace(self):
        for arithmetic, result in (("sfpmul(x,y)", 12), ("sfpadd(x,y)", 7),
                                   ("sfpmad(x,y,y)", 16), ("sfpmad(x,sfpmul(y,y),y)", 52)):
            for extra, suffix, copies in (("", "", False),
                                           (",preserve=(0,)", "", True),
                                           (",outputs={'x':7}", "    return x\n", True),
                                           ("", "    sfpstore(old,2)\n", True)):
                source = (f"@sfpu(inputs={{'x':0,'y':1,'test':2}}{extra})\ndef f(x,y,test):\n"
                          "    old=x\n    with sfpsetcc(test,cc='lt0'):\n"
                          f"        x={arithmetic}\n    sfpstore(x,0)\n" + suffix)
                code = Compiler().compile(source)
                self.assertEqual("TTI_SFPMOV" in code, copies)
                for test in (-1, 0, 1):
                    regs = {0: 3, 1: 4, 2: test, 9: 0, 10: 1}
                    dst = {}
                    active = True
                    for op, operands in re.findall(r"TTI_(\w+)\(([^)]*)\)", code):
                        a = [int(x.strip().removesuffix("u")) for x in operands.split(",")]
                        if op == "SFPENCC":
                            active = True
                        elif op == "SFPSETCC":
                            active = regs[a[1]] < 0
                        elif op == "SFPMOV":
                            if active or a[3] == 2:
                                regs[a[2]] = regs[a[1]]
                        elif op in ("SFPMUL", "SFPADD", "SFPMAD"):
                            if active:
                                regs[a[3]] = (regs[a[1]] + regs[a[2]] if op == "SFPADD" else
                                              regs[a[0]] * regs[a[1]] + regs[a[2]])
                        elif op == "SFPSTORE":
                            if active:
                                dst[a[3]] = regs[a[0]]
                        else:
                            self.fail(op)
                    self.assertEqual(dst[0], result if test < 0 else 3)
                    if "sfpstore(old" in suffix:
                        self.assertEqual(dst[2], 3)
                    if "preserve" in extra:
                        self.assertEqual(regs[0], 3)
                    if "outputs" in extra:
                        self.assertEqual(regs[7], dst[0])

    def test_predicate_rejections(self):
        for body in ('with sfpiadd(x,0,cc="gte0"):\n        y=sfpmul(x,x)\n    sfpstore(y,0)',
                     'with sfpiadd(x,0,cc="gte0"):\n        with sfpiadd(x,0,cc="gte0"):\n            x=sfploadi(0x3f800000)',
                     'x=sfpiadd(x,0,cc="gte0")'):
            with self.subTest(body=body), self.assertRaises(CompileError):
                Compiler().compile('@sfpu(inputs={"x":0})\ndef f(x):\n    '+body)

    def test_sfploadi_lowering(self):
        for bits in (0x80000000, 0x00010000, 0x7f800000, 0x7fc10000,
                     0xffff0000, 1, 0x3f800001, 0xdeadbeef, 0xffffffff):
            code = Compiler().compile(f'@sfpu(outputs={{"x":3}})\ndef f():\n'
                                      f'    x=sfploadi({bits})\n    return x')
            loads = re.findall(r'TTI_SFPLOADI\((\d+)u, (\d+)u, (\d+)u\)', code)
            self.assertEqual(len(loads), 2 if bits & 65535 else 1)
            value = 0xdeadbeef
            for reg, mode, imm in loads:
                self.assertEqual(int(reg), 3)
                mode, imm = int(mode), int(imm)
                if mode == 0:
                    value = imm << 16
                elif mode == 8:
                    value = (imm << 16) | (value & 65535)
                else:
                    self.assertEqual(mode, 10)
                    value = (value & 0xffff0000) | imm
            self.assertEqual(value, bits)
        for bits, reg in ((0, 9), (0x3f800000, 10), (0x3f56594b, 8)):
            source = f'@sfpu()\ndef f():\n    x=sfploadi({bits})\n    sfpstore(x,0)'
            code = Compiler().compile(source)
            self.assertEqual(code.count('TTI_'), 1)
            self.assertIn(f'TTI_SFPSTORE({reg}u, 3u, 0u, 0u)', code)
            code = Compiler().compile(f'@sfpu()\ndef f():\n    x=sfpmuli(sfploadi({bits}),0x4000)\n    sfpstore(x,0)')
            self.assertIn(f'TTI_SFPMOV(0u, {reg}u, 0u, 0u)', code)
            self.assertIn('TTI_SFPMULI(16384u, 0u, 0u)', code)
        with self.assertRaises(CompileError):
            Compiler().compile('@sfpu()\ndef f():\n    x=bits32(0)')
        with self.assertRaises(CompileError):
            Compiler().compile('@sfpu()\ndef f():\n    x=sfploadi(0x3f80,fmt="bf16")')

    def test_raw_bits_and_tied_operand(self):
        code = Compiler().compile('''@sfpu(inputs={"a":0,"e":1})
def f(a,e):
    b=sfploadi(0xdeadbeef)
    sfpstore(b,0)
    x=sfpsetexp(a,e)
    sfpstore(x,2)
''')
        self.assertIn("TTI_SFPLOADI(2u, 8u, 57005u)", code)
        self.assertIn("TTI_SFPLOADI(2u, 10u, 48879u)", code)
        self.assertIn("TTI_SFPSETEXP(0u, 0u, 1u, 0u)", code)

    def test_nested_and_boundaries(self):
        result = Compiler().compile('''@sfpu(inputs={"a":0,"c":1,"g":2,"e":3}, outputs={"out":0}, preserve=(7,))
def f(a,c,g,e):
    out=sfpmad(c,g,sfpmul(a,e))
    return out
''')
        self.assertIn("TTI_SFPMUL(0u, 3u, 9u, 0u, 0u)", result)
        self.assertIn("TTI_SFPMAD(1u, 2u, 0u, 0u, 0u)", result)
        self.assertIn("Written LRegs (including outputs): [0]", result)
        with self.assertRaisesRegex(CompileError, "allocation failed"):
            Compiler().compile('''@sfpu(inputs={"a":0}, outputs={"out":0})
def f(a):
    out=sfpload(0)
    sfpstore(a,2)
    return out
''')

    def test_cpp_macros(self):
        header = Path(__file__).resolve().parents[1] / "src/ckernel_ops.h"
        production = header.parent / "kernels.sfpu"
        source = (f'#include <cstdint>\n#include "{header}"\n#undef INSTRUCTION_WORD\n'
                  '#define INSTRUCTION_WORD(x) static_assert((unsigned(x) & 0xff000000u) != 0)\n'
                  'static inline void tensix_instruction(uint32_t) {}\n'
                  'static inline uint32_t pack_u16(uint32_t lo, uint32_t hi) { return (lo & 65535u) | (hi << 16); }\n'
                  + Compiler().compile(EXAMPLES)
                  + Compiler().compile(production.read_text(), str(production))
                  + '\nvoid instantiate() { sfpu_attention_scale_body(0x3f800000u); sfpu_attention_accumulate_body(); '
                    'sfpu_attention_normalize_body<28>(0x3f800000u); '
                    'sfpu_swiglu_body<0>(); sfpu_swiglu_body<2>(); }\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.cpp"
            path.write_text(source)
            result = subprocess.run(["g++", "-std=gnu++20", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_example_dataflow(self):
        # Small exact integers isolate allocation/emission from floating-point numerics.
        blocks = Compiler().compile(EXAMPLES).split("static inline void ")[1:]
        for block in blocks:
            name = block.split("(")[0]
            dst = {i: i + 2 for i in range(16)}
            regs = {i: i + 3 for i in range(15)}
            regs[9], regs[10] = 0, 1
            expected_dst = dst.copy()
            expected_regs = {}
            if name == "residual_add":
                expected_dst.update({0: 8, 2: 12})
            elif name == "scale_weight":
                expected_dst.update({0: 2 * 9 * 6, 2: 4 * 9 * 8})
            elif name == "rope":
                expected_dst.update({0: 2 * 10 + 6 * 14, 2: 4 * 12 + 8 * 16})
            elif name == "weighted_value":
                expected_dst.update({0: 2 + 9 * 7, 2: 4 + 10 * 7})
                expected_regs.update({6: 9, 7: 10})
            elif name == "squares":
                expected_regs.update({4: 7 + 2 * 2, 5: 8 + 4 * 4})
            for op, operands in re.findall(r"TTI_(\w+)\(([^)]*)\)", block):
                a = [int(x.strip().removesuffix("u")) for x in operands.split(",")]
                if op == "SFPLOAD":
                    regs[a[0]] = dst[a[3]]
                elif op == "SFPSTORE":
                    dst[a[3]] = regs[a[0]]
                elif op == "SFPADD":
                    regs[a[3]] = regs[a[1]] + regs[a[2]]
                elif op == "SFPMUL":
                    regs[a[3]] = regs[a[0]] * regs[a[1]]
                else:
                    self.assertEqual(op, "SFPMAD")
                    regs[a[3]] = regs[a[0]] * regs[a[1]] + regs[a[2]]
            with self.subTest(kernel=name):
                self.assertEqual(dst, expected_dst)
                for reg, value in expected_regs.items():
                    self.assertEqual(regs[reg], value)

    def test_examples(self):
        result = Compiler().compile(EXAMPLES)
        self.assertEqual(result.count("static inline void"), 5)
        self.assertIn("TTI_SFPMAD(0u, 0u, 4u, 4u, 0u)", result)

    def test_exact_emission(self):
        result = Compiler().compile('@sfpu()\ndef add():\n    sfpstore(sfpadd(sfpload(0), sfpload(2)), 4)')
        instructions = [line.strip().split(" //")[0] for line in result.splitlines() if "TTI_" in line]
        self.assertEqual(instructions, ["TTI_SFPLOAD(0u, 3u, 0u, 0u);", "TTI_SFPLOAD(1u, 3u, 0u, 2u);",
                                        "TTI_SFPADD(10u, 0u, 1u, 0u, 0u);", "TTI_SFPSTORE(0u, 3u, 0u, 4u);"])

    def test_memory_addrmod(self):
        result = Compiler().compile('@sfpu()\ndef f():\n    x = sfpload(0, addrmod=1)\n    sfpstore(x, 2, addrmod=3)')
        self.assertIn("TTI_SFPLOAD(0u, 3u, 1u, 0u)", result)
        self.assertIn("TTI_SFPSTORE(0u, 3u, 3u, 2u)", result)
        for value in ("-1", "4", "True", "1.0", "x"):
            for statement in (f"x = sfpload(0, addrmod={value})", f"sfpstore(sfploadi(0x3f800000), 0, addrmod={value})"):
                with self.assertRaises(CompileError):
                    Compiler().compile('@sfpu()\ndef f():\n    ' + statement)

    def test_template_addrmod(self):
        code = Compiler().compile("@sfpu(addrmods=('step',))\ndef f():\n    sfpstore(ZERO,0,addrmod=step)")
        self.assertIn("template<uint32_t step>", code)
        self.assertIn('static_assert(step < 4u, "SFPU AddrMod out of range")', code)
        for body in ("step=sfpload(0)", "x=sfpload(step)", "x=sfpmul(step,sfploadi(0x3f800000))",
                     "setrwc(dst=step)", "sfpstore(ZERO,0,addrmod=step+1)"):
            with self.subTest(body=body), self.assertRaises(CompileError):
                Compiler().compile("@sfpu(addrmods=('step',))\ndef f():\n    " + body)
        for contract in ("addrmods='step'", "addrmods=('step','step')",
                         "addrmods=('step',),templates=('step',)",
                         "addrmods=('step',),offsets=('step',)", "addrmods=('ZERO',)"):
            with self.subTest(contract=contract), self.assertRaises(CompileError):
                Compiler().compile("@sfpu(" + contract + ")\ndef f():\n    sfpstore(ZERO,0)")

    def test_constants_keywords(self):
        result = Compiler().compile('@sfpu(inputs={"k": 11})\ndef f(k):\n    sfpstore(offset=0, value=sfpmul(b=k, a=sfpload(fmt="bf16", offset=2)))')
        self.assertIn("TTI_SFPLOAD(0u, 2u, 0u, 2u)", result)
        self.assertIn("TTI_SFPMUL(0u, 11u, 9u, 0u, 0u)", result)

    def test_rejections(self):
        sources = ["import os", '@sfpu()\ndef f():\n    x = sfptransp()',
                   '@sfpu()\ndef f():\n    sfpnop()',
                   '@sfpu()\ndef f():\n    x = sfpshft2(ZERO)',
                   '@sfpu()\ndef f():\n    x = sfpshft2(ZERO,mode="copy4")',
                   '@sfpu()\ndef f():\n    x = sfpshft2(ZERO,mode=3)',
                   '@sfpu()\ndef f():\n    x = sfpload(0, fmt="bad")',
                   '@sfpu()\ndef f():\n    sfpstore(missing, 0)',
                   '@sfpu(preserve=(0,1,2,3,4,5,6,7))\ndef f():\n    x = sfpload(0)',
                   '@sfpu(outputs={"x":0}, preserve=(0,))\ndef f():\n    x=sfpload(0)\n    return x',
                   '@sfpu()\ndef f():\n    x=sfpload(True)',
                   '@sfpu()\ndef f():\n    for x in range(2):\n        sfpstore(ZERO,0)']
        for source in sources:
            with self.subTest(source=source), self.assertRaises(CompileError):
                Compiler().compile(source)

    def test_pressure(self):
        source = "@sfpu()\ndef f():\n" + "".join(f"    x{i}=sfpload({i})\n" for i in range(9))
        source += "".join(f"    sfpstore(x{i},{i})\n" for i in range(9))
        with self.assertRaisesRegex(CompileError, "allocation failed"):
            Compiler().compile(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return not unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()
    if args.examples:
        print(EXAMPLES, end="")
        return 0
    if args.input is None:
        parser.error("input .sfpu file required")
    try:
        result = Compiler().compile(args.input.read_text(), str(args.input))
        if args.output:
            args.output.write_text(result)
        else:
            print(result, end="")
    except (CompileError, OSError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
