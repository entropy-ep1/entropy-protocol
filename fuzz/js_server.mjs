// The second implementation behind a JSON-lines pipe, for differential fuzzing (fuzz/targets.py): one request per
// line {"fn", "args"}, one answer per line {"ok": value} or {"error": message}. Values are given in the plain shape
// fuzz/targets.py:_plain gives for Python's, so the two can be compared as JSON.
import readline from "node:readline";
import * as ep1 from "../impl-js/ep1.mjs";

function plain(v) {
  if (v === undefined) return { $undefined: true };
  if (v === null || typeof v === "boolean" || typeof v === "string") return v;
  if (typeof v === "bigint") return { $int: v.toString() };
  if (typeof v === "number") {
    if (Number.isNaN(v)) return { $float: "nan" };
    if (Number.isInteger(v) && !Object.is(v, -0) && Math.abs(v) < 2 ** 53) return { $int: String(v) };
    const b = Buffer.alloc(8); b.writeDoubleBE(v); return { $float: b.toString("hex") };
  }
  if (Buffer.isBuffer(v)) return { $bytes: v.toString("hex") };
  if (Array.isArray(v)) return v.map(plain);
  if (v instanceof Map) return { $map: [...v.entries()].map(([k, x]) => [plain(k), plain(x)]) };
  if (v instanceof ep1.Tag) return { $tag: String(v.tag), value: plain(v.value) };
  return { $other: typeof v };
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  let out;
  try {
    const { fn, args } = JSON.parse(line);
    if (fn === "canonicalize") out = { ok: ep1.canonicalize(args[0], args[1]) };
    else if (fn === "segment") out = { ok: ep1.segment(args[0], args[1]) };
    else if (fn === "plainNormalize") out = { ok: ep1.plainNormalize(args[0]) };
    else if (fn === "decode") out = { ok: plain(ep1.decode(Buffer.from(args[0], "hex"))) };
    else if (fn === "binomialTail") out = { ok: ep1.binomialTail(args[0], args[1]) };
    else out = { error: `unknown function ${fn}` };
  } catch (e) {
    out = { error: String((e && e.message) || e) };
  }
  process.stdout.write(JSON.stringify(out) + "\n");
});
