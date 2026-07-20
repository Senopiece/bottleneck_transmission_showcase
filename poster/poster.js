import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import pptxgen from "pptxgenjs";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(DIR);
const WIDTH = 2384;
const HEIGHT = 3370;

const C = {
  page: "#DCE6F7",
  white: "#FFFFFF",
  ink: "#08090B",
  muted: "#5D6572",
  line: "#B9C5D8",
  panel: "#F4F6FA",
  blue: "#4C52F7",
  cyan: "#5AA9E6",
  amber: "#F2B84B",
  red: "#D64747",
  gray: "#ADB5C1",
  dark: "#242833",
};

const logoData = fs.readFileSync(path.join(DIR, "erko.jpg")).toString("base64");
const opticalCapture3dData = fs.readFileSync(path.join(DIR, "assets/optical_capture_3d.png")).toString("base64");
const message = JSON.parse(
  fs.readFileSync(path.join(ROOT, "receiver/datasets/raw/expected/default_creeper_8x8.json"), "utf8"),
);

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function line(x1, y1, x2, y2, stroke = C.ink, width = 3, extra = "") {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}" ${extra}/>`;
}

function rect(x, y, w, h, fill = C.white, stroke = "none", sw = 0, rx = 0, extra = "") {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" ${extra}/>`;
}

function circle(cx, cy, r, fill = C.white, stroke = C.ink, sw = 3, extra = "") {
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" ${extra}/>`;
}

function arrow(x1, y1, x2, y2, color = C.ink, width = 3) {
  return `${line(x1, y1, x2, y2, color, width, 'marker-end="url(#arrow)"')}`;
}

function wrapWords(text, maxChars) {
  const words = text.split(/\s+/);
  const lines = [];
  let current = "";
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > maxChars && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  }
  if (current) lines.push(current);
  return lines;
}

function text(x, y, value, size, options = {}) {
  const {
    weight = 400,
    fill = C.ink,
    anchor = "start",
    family = "Arial, Helvetica, sans-serif",
    italic = false,
    letterSpacing = 0,
  } = options;
  return `<text x="${x}" y="${y}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" font-style="${italic ? "italic" : "normal"}" letter-spacing="${letterSpacing}">${esc(value)}</text>`;
}

function textBlock(x, y, value, size, maxChars, lineHeight = 1.22, options = {}) {
  const lines = Array.isArray(value) ? value : wrapWords(value, maxChars);
  const {
    weight = 400,
    fill = C.ink,
    anchor = "start",
    family = "Arial, Helvetica, sans-serif",
  } = options;
  return `<text x="${x}" y="${y}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${lines
    .map((ln, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : size * lineHeight}">${esc(ln)}</tspan>`)
    .join("")}</text>`;
}

function bulletList(x, y, items, size, maxChars, gap = 12) {
  let out = "";
  let yy = y;
  for (const item of items) {
    const lines = wrapWords(item, maxChars);
    out += circle(x + 5, yy - size * 0.27, 4.5, C.ink, "none", 0);
    out += textBlock(x + 24, yy, lines, size, maxChars, 1.18);
    yy += lines.length * size * 1.18 + gap;
  }
  return out;
}

function section(x, y, w, h, titleLabel) {
  const tabW = Math.max(250, titleLabel.length * 31 + 45);
  return `${rect(x, y, w, h, C.white)}${rect(x, y - 54, tabW, 64, C.white)}${text(x + 10, y - 5, titleLabel, 52, { weight: 700 })}`;
}

function triangle(cx, cy, s, fill, stroke, sw = 4) {
  const h = s * 0.88;
  const pts = `${cx},${cy - h / 2} ${cx - s / 2},${cy + h / 2} ${cx + s / 2},${cy + h / 2}`;
  return `<polygon points="${pts}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round"/>`;
}

function messageGrid(x, y, cell, rows = message.rows, opacity = 1) {
  let out = "";
  rows.forEach((row, r) => {
    [...row].forEach((bit, c) => {
      const fill = bit === "1" ? C.ink : C.white;
      out += rect(x + c * cell, y + r * cell, cell - 3, cell - 3, fill, C.line, 1.5, 2, `opacity="${opacity}"`);
    });
  });
  return out;
}

function beliefGrid(x, y, cell, revealedFraction, rows = message.rows) {
  let out = "";
  rows.forEach((row, r) => {
    [...row].forEach((bit, c) => {
      const xx = x + c * cell;
      const yy = y + r * cell;
      const index = r * row.length + c;
      const revealRank = (index * 37 + 11) % 64;
      const revealed = revealRank < Math.round(revealedFraction * 64);
      out += rect(xx, yy, cell - 2, cell - 2, revealed ? (bit === "1" ? C.ink : C.white) : "#AAB1BC", C.line, 1, 1.5);
    });
  });
  return out;
}

function ledPattern(x, y, scale = 1, bits = "10101") {
  const out = [];
  const sy = 34 * scale;
  out.push(rect(x, y - 46 * scale, 500 * scale, 92 * scale, C.dark, C.line, 2.5 * scale, 18 * scale));
  out.push(rect(x + 28 * scale, y - 16 * scale, sy, sy, C.white, C.white, 0, 2 * scale));
  for (let i = 0; i < 5; i++) {
    const on = bits[i] === "1";
    out.push(circle(x + (120 + i * 65) * scale, y, 17 * scale, on ? C.cyan : "#26394F", on ? C.white : C.cyan, 3 * scale));
  }
  out.push(triangle(x + 452 * scale, y, 40 * scale, C.white, C.white, 2.5 * scale));
  return out.join("");
}

function figureOptical(x, y, w, h) {
  let out = rect(x, y, w, h, C.panel, C.line, 2, 10);
  out += text(x + 28, y + 52, "2  Optical capture", 34, { weight: 700 });
  out += text(x + 28, y + 88, "Constrained geometry, soft output", 23, { fill: C.muted });

  // A single coherent 3D render: the smaller target and larger phone face each other.
  out += `<image x="${x + 26}" y="${y + 105}" width="660" height="258" href="data:image/png;base64,${opticalCapture3dData}" preserveAspectRatio="xMidYMid meet"/>`;
  const barX = x + 60;
  const barY = y + 355;
  out += text(barX, barY, "Soft LED likelihoods", 24, { weight: 700 });
  [0.87, 0.16, 0.91, 0.21, 0.84].forEach((v, i) => {
    const yy = barY + 35 + i * 32;
    out += text(barX, yy + 18, `L${i + 1}`, 18, { fill: C.muted });
    out += rect(barX + 40, yy, 310, 20, "#DEE4EC", "none", 0, 10);
    out += rect(barX + 40, yy, 310 * v, 20, v > 0.5 ? C.blue : C.gray, "none", 0, 10);
  });
  out += textBlock(x + 430, y + 385, ["The CV front end estimates", "one global pose and returns", "five soft likelihoods —", "not premature hard bits."], 20, 27, 1.23);
  out += text(x + w / 2, y + h - 20, "Figure 2. CV returns soft five-channel observations.", 20, { anchor: "middle", fill: C.muted });
  return out;
}

function figureEncoder(x, y, w, h) {
  let out = rect(x, y, w, h, C.panel, C.line, 2, 10);
  out += text(x + 28, y + 52, "1  Rateless encoder", 34, { weight: 700 });
  out += text(x + 28, y + 88, "LDPC precode + ongoing sparse LDGM measurements", 23, { fill: C.muted });
  out += messageGrid(x + 25, y + 132, 22);
  ["x0", "x1", "x2", "..."].forEach((label, index) => {
    const bit = message.rows[0][index];
    out += text(x + 34.5 + index * 22, y + 146, label, 7.5, {
      anchor: "middle",
      weight: 700,
      fill: bit === "1" ? C.white : C.muted,
    });
  });
  out += text(x + 111, y + 325, "8 x 8 message", 18, { anchor: "middle", fill: C.muted });
  out += arrow(x + 212, y + 218, x + 242, y + 218, C.ink, 2.5);
  out += rect(x + 252, y + 159, 125, 118, C.white, C.blue, 4, 8);
  out += text(x + 314.5, y + 205, "LDPC", 27, { anchor: "middle", weight: 700, fill: C.blue });
  out += text(x + 314.5, y + 237, "precode", 21, { anchor: "middle" });
  out += arrow(x + 389, y + 218, x + 419, y + 218, C.ink, 2.5);
  out += rect(x + 429, y + 159, 150, 118, C.white, C.blue, 4, 8);
  out += text(x + 504, y + 205, "LDGM", 27, { anchor: "middle", weight: 700, fill: C.blue });
  out += text(x + 504, y + 237, "fountain encoder", 18, { anchor: "middle" });
  out += arrow(x + 591, y + 218, x + 621, y + 218, C.ink, 2.5);
  const streamBits = "101010011011000";
  [...streamBits].forEach((bit, bitIndex) => {
    const group = Math.floor(bitIndex / 5);
    const sx = x + 632 + bitIndex * 10 + group * 6;
    out += rect(sx, y + 208, 7, 20, bit === "1" ? C.blue : C.white, C.blue, 1.5, 1.5);
  });
  [[632, 679], [688, 735], [744, 791]].forEach(([start, end]) => {
    out += line(x + start, y + 239, x + end, y + 239, C.blue, 1.5);
    out += line(x + start, y + 235, x + start, y + 243, C.blue, 1.5);
    out += line(x + end, y + 235, x + end, y + 243, C.blue, 1.5);
    out += text(x + (start + end) / 2, y + 258, "5 bits", 11, { anchor: "middle", fill: C.muted });
  });
  out += text(x + 806, y + 221, "...", 20, { anchor: "middle", weight: 700, fill: C.blue });
  out += text(x + 711, y + 292, "infinite stream", 17, { anchor: "middle", weight: 700, fill: C.blue });
  const ty = y + 425;
  out += text(x + 131, ty, "Preamble", 22, { anchor: "end", weight: 700 });
  ["00000", "01010", "10101", "11111"].forEach((b, i) => {
    out += rect(x + 145 + i * 112, ty - 28, 98, 42, C.white, C.line, 2, 5);
    out += text(x + 194 + i * 112, ty, b, 18, { anchor: "middle", family: "Courier New, monospace", weight: 700 });
  });
  out += arrow(x + 596, ty - 8, x + 626, ty - 8, C.blue, 2.5);
  out += text(x + 642, ty - 1, "infinite stream", 17, { fill: C.blue, weight: 700 });
  out += text(x + 194, ty + 42, "detect start + infer symbol timing", 17, { anchor: "middle", fill: C.muted });
  out += text(x + w / 2, y + h - 20, "Figure 1. LDPC protects the message; sparse LDGM packets continue ratelessly.", 19, { anchor: "middle", fill: C.muted });
  return out;
}

function figureDecoder(x, y, w, h) {
  let out = rect(x, y, w, h, C.panel, C.line, 2, 10);
  out += text(x + 28, y + 52, "3  Online soft BP", 34, { weight: 700 });
  out += text(x + 28, y + 88, "Bounded graph, residual scheduling, predictive guard", 22, { fill: C.muted });

  // Permanent LDPC checks and codeword variables.
  out += text(x + 25, y + 124, "32 permanent LDPC checks", 17, { weight: 700, fill: C.blue });
  const vx = [x + 72, x + 170, x + 268, x + 366, x + 464, x + 562];
  const vy = y + 225;
  vx.forEach((xx, i) => {
    out += circle(xx, vy, 21, C.white, C.blue, 3);
    out += text(xx, vy + 6, `x${i}`, 16, { anchor: "middle", weight: 700, fill: C.blue });
  });
  const topFs = [x + 120, x + 316, x + 512];
  topFs.forEach((xx, i) => {
    out += rect(xx - 15, y + 138, 30, 30, C.dark, C.dark, 0, 4);
    [i * 2, i * 2 + 1, (i * 2 + 2) % 6].forEach((idx) => { out += line(xx, y + 168, vx[idx], vy - 21, C.line, 2); });
  });

  // Circular measurement-factor bank. New factors enter at right; oldest slot leaves at left.
  const bankY = y + 290;
  out += rect(x + 45, bankY, 582, 112, "#EEF1FF", C.blue, 2, 8);
  const obs = [
    { x: x + 177, links: [0, 2, 5] },
    { x: x + 274, links: [1, 3, 4] },
    { x: x + 371, links: [0, 1, 4] },
    { x: x + 468, links: [2, 3, 5] },
    { x: x + 565, links: [0, 3, 5] },
  ];
  obs.forEach((f) => {
    f.links.forEach((idx) => { out += line(f.x, bankY + 56, vx[idx], vy + 21, C.line, 2); });
    out += rect(f.x - 15, bankY + 41, 30, 30, C.blue, C.blue, 0, 4);
  });
  out += rect(x + 183, bankY + 5, 306, 25, "#EEF1FF");
  out += text(x + 336, bankY + 24, "bounded bank: 120 newest LDGM factors", 17, { anchor: "middle", weight: 700, fill: C.blue });
  out += rect(x + 647, bankY + 41, 30, 30, C.blue, C.blue, 0, 4);
  out += arrow(x + 642, bankY + 56, x + 612, bankY + 56, C.blue, 2.5);
  out += text(x + 662, bankY + 97, "new", 15, { anchor: "middle", weight: 700, fill: C.blue });
  out += rect(x + 65, bankY + 41, 30, 30, C.gray, C.gray, 0, 4, 'opacity="0.55"');
  out += arrow(x + 60, bankY + 56, x + 30, bankY + 56, C.gray, 2.5);
  out += text(x + 80, bankY + 97, "oldest out", 14, { anchor: "middle", fill: C.muted });
  out += text(x + 336, bankY + 99, "full bank: incoming factor overwrites oldest slot", 14, { anchor: "middle", fill: C.muted });

  // Indexed residual queue and one bounded work unit.
  const pumpY = y + 414;
  out += rect(x + 20, pumpY, 649, 200, C.white, C.line, 2, 8);
  out += text(x + 38, pumpY + 29, "Residual BP pump", 19, { weight: 700 });
  out += text(x + 225, pumpY + 29, "every 16 ms, independent of packet arrival", 15, { fill: C.muted });
  out += rect(x + 40, pumpY + 48, 155, 62, "#F7F8FB", C.line, 2, 6);
  out += text(x + 117, pumpY + 68, "indexed max-queue", 14, { anchor: "middle", weight: 700 });
  ["f17  delta .84", "f6   delta .41", "f..."].forEach((label, i) => {
    out += text(x + 55, pumpY + 85 + i * 12, label, 10.5, { family: "Courier New, monospace", fill: i === 0 ? C.blue : C.muted });
  });
  const flowY = pumpY + 79;
  out += arrow(x + 202, flowY, x + 228, flowY, C.blue, 2.5);
  out += rect(x + 236, flowY - 17, 34, 34, C.blue, C.blue, 0, 4);
  out += text(x + 253, pumpY + 123, "selected factor", 12, { anchor: "middle", fill: C.muted });
  out += arrow(x + 277, flowY - 8, x + 322, flowY - 8, C.ink, 2.2);
  out += arrow(x + 322, flowY + 8, x + 277, flowY + 8, C.ink, 2.2);
  [x + 345, x + 390, x + 435].forEach((xx) => { out += circle(xx, flowY, 14, C.white, C.blue, 2.5); });
  out += text(x + 390, pumpY + 123, "variables", 12, { anchor: "middle", fill: C.muted });
  out += arrow(x + 456, flowY, x + 490, flowY, C.ink, 2.2);
  out += rect(x + 506, flowY - 12, 24, 24, C.blue, C.blue, 0, 3);
  out += rect(x + 548, flowY - 12, 24, 24, C.dark, C.dark, 0, 3);
  out += rect(x + 590, flowY - 12, 24, 24, C.blue, C.blue, 0, 3);
  out += text(x + 560, pumpY + 123, "neighboring LDGM + LDPC factors", 12, { anchor: "middle", fill: C.muted });
  out += text(x + 405, pumpY + 143, "recompute residuals and update queue priorities", 13, { anchor: "middle", weight: 700, fill: C.blue });
  out += arrow(x + 614, pumpY + 152, x + 195, pumpY + 152, C.blue, 2.2);
  out += rect(x + 40, pumpY + 164, 280, 28, "#EEF1FF", "none", 0, 5);
  out += text(x + 180, pumpY + 183, "24 updates / pump  |  bounded work", 13, { anchor: "middle", weight: 700, fill: C.muted });
  out += rect(x + 330, pumpY + 164, 319, 28, "#EEF1FF", "none", 0, 5);
  out += text(x + 489.5, pumpY + 183, "posterior recomputed after every pump", 13, { anchor: "middle", weight: 700, fill: C.blue });

  // Predictive consistency guard mirrors FountainDecoderController.
  const guardY = y + 626;
  out += rect(x + 20, guardY, 649, 158, C.white, C.line, 2, 8);
  out += text(x + 38, guardY + 29, "Inconsistent-stream guard", 19, { weight: 700 });
  out += text(x + 649, guardY + 29, "separate from BP convergence", 15, { anchor: "end", fill: C.muted });

  const guardFlowY = guardY + 74;
  out += rect(x + 63, guardY + 49, 122, 48, "#EEF1FF", C.blue, 2, 5);
  out += textBlock(x + 124, guardY + 68, ["best posterior", "after >=25%"], 13, 18, 1.12, { anchor: "middle", weight: 700, fill: C.blue });
  out += arrow(x + 194, guardFlowY, x + 220, guardFlowY, C.ink, 2.5);
  out += rect(x + 229, guardY + 49, 128, 48, C.white, C.line, 2, 5);
  out += textBlock(x + 293, guardY + 68, ["predict incoming", "soft packet"], 13, 18, 1.12, { anchor: "middle", weight: 700 });
  out += arrow(x + 366, guardFlowY, x + 392, guardFlowY, C.ink, 2.5);
  out += rect(x + 401, guardY + 49, 136, 48, C.white, C.line, 2, 5);
  out += textBlock(x + 469, guardY + 68, ["log-compatibility", "bad = max(-log, 0)"], 12, 21, 1.12, { anchor: "middle", weight: 700 });
  out += arrow(x + 546, guardFlowY, x + 572, guardFlowY, C.red, 2.5);
  out += circle(x + 606, guardFlowY, 20, C.white, C.red, 4);
  out += line(x + 592, guardFlowY - 14, x + 620, guardFlowY + 14, C.red, 4);
  out += line(x + 620, guardFlowY - 14, x + 592, guardFlowY + 14, C.red, 4);

  out += text(x + 344, guardY + 119, "hit: packet bad >= 0.15 AND sum(last 4 bad) >= 2.0", 14, { anchor: "middle", family: "Courier New, monospace", fill: C.muted });
  out += rect(x + 60, guardY + 126, 569, 24, "#FCEBEC", "none", 0, 5);
  out += line(x + 344, guardY + 130, x + 344, guardY + 146, "#F5CACC", 1.5);
  out += text(x + 128, guardY + 143, "first hit", 14, { anchor: "middle", weight: 700, fill: C.red });
  out += arrow(x + 165, guardY + 138, x + 195, guardY + 138, C.red, 2.2);
  out += text(x + 245, guardY + 143, "skip packet", 14, { anchor: "middle", weight: 700, fill: C.red });
  out += text(x + 405, guardY + 143, "second consecutive hit", 14, { anchor: "middle", weight: 700, fill: C.red });
  out += arrow(x + 492, guardY + 138, x + 522, guardY + 138, C.red, 2.2);
  out += text(x + 566, guardY + 143, "reject", 14, { anchor: "middle", weight: 700, fill: C.red });

  out += text(x + w / 2, y + h - 20, "Figure 3. Fixed graph memory and residual scheduling bound online BP work.", 18, { anchor: "middle", fill: C.muted });
  return out;
}

function resultPhone(x, y, w, h) {
  let out = rect(x, y, w, h, C.panel, C.line, 2, 10);
  // A slim phone with enough screen area to make both optical input and soft output legible.
  out += rect(x + 42, y + 24, 224, 424, "#11141B", C.ink, 4, 34);
  out += rect(x + 54, y + 42, 200, 382, C.dark, "none", 0, 25);
  out += rect(x + 132, y + 50, 44, 8, "#777F8D", "none", 0, 4);
  const phonePatternCenterX = x + 154;
  out += `<g transform="translate(${phonePatternCenterX} 0) scale(0.9 1) translate(${-phonePatternCenterX} 0)">${ledPattern(x + 69, y + 126, 0.34, "10101")}</g>`;
  out += beliefGrid(x + 78, y + 204, 19, 0.78);
  out += rect(x + 78, y + 382, 152, 7, "#4B5260", "none", 0, 4);
  out += rect(x + 78, y + 382, 112, 7, C.blue, "none", 0, 4);
  out += text(x + 154, y + 414, "soft posterior", 16, { anchor: "middle", fill: C.white });

  out += text(x + 310, y + 66, "Receiver behavior", 28, { weight: 700 });
  out += text(x + 310, y + 112, "A short camera gap becomes erasures", 21, { weight: 700 });
  out += line(x + 310, y + 156, x + 842, y + 156, "#DDE2EB", 9);
  out += line(x + 310, y + 156, x + 500, y + 156, C.blue, 9);
  out += line(x + 565, y + 156, x + 842, y + 156, C.blue, 9);
  out += line(x + 510, y + 156, x + 555, y + 156, C.gray, 9, 'stroke-dasharray="9 7"');
  out += text(x + 310, y + 192, "received evidence is retained", 18, { fill: C.blue, weight: 700 });

  out += text(x + 310, y + 265, "More packets resolve additional bits", 21, { weight: 700 });
  const gridY = y + 294;
  const gridXs = [x + 330, x + 512, x + 694];
  [0.16, 0.52, 1.0].forEach((revealedFraction, i) => {
    out += beliefGrid(gridXs[i], gridY, 10, revealedFraction);
  });
  out += arrow(x + 426, y + 333, x + 486, y + 333, C.blue, 2.2);
  out += arrow(x + 608, y + 333, x + 668, y + 333, C.blue, 2.2);
  out += text(gridXs[0] + 39, y + 401, "few bits resolve", 16, { anchor: "middle", fill: C.muted });
  out += text(gridXs[1] + 39, y + 401, "more bits resolve", 16, { anchor: "middle", fill: C.muted });
  out += text(gridXs[2] + 39, y + 401, "decoded", 16, { anchor: "middle", fill: C.muted });
  out += text(x + w / 2, y + h - 18, "Figure 4. Information accumulates continuously and the message gradually appears.", 20, { anchor: "middle", fill: C.muted });
  return out;
}

function buildPosterSvg() {
  const s = [];
  s.push(`<svg xmlns="http://www.w3.org/2000/svg" width="841mm" height="1189mm" viewBox="0 0 ${WIDTH} ${HEIGHT}">`);
  s.push(`<defs>
    <marker id="arrow" markerWidth="5" markerHeight="5" refX="4.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="context-stroke"/></marker>
    <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="8"/></filter>
  </defs>`);
  s.push(rect(0, 0, WIDTH, HEIGHT, C.page));

  // Header and identity.
  s.push(text(45, 67, "Skoltech Industrial Immersion", 31, { weight: 700, fill: "#8FAFE0" }));
  s.push(text(WIDTH - 50, 67, "2026", 31, { weight: 700, fill: "#8FAFE0", anchor: "end" }));
  s.push(textBlock(45, 205, ["Soft Rateless Optical", "Communication with a", "Mobile Camera Receiver"], 100, 31, 0.98, { weight: 700 }));
  s.push(text(1995, 150, "In collaboration with", 25));
  s.push(text(1995, 183, "ERKO", 29, { weight: 700 }));
  s.push(rect(1995, 205, 290, 290, C.white));
  s.push(`<image x="2012" y="222" width="256" height="256" href="data:image/jpeg;base64,${logoData}" preserveAspectRatio="xMidYMid meet"/>`);

  s.push(text(45, 690, "Student", 27, { weight: 700 }));
  s.push(text(45, 727, "Vitaly Makhonin", 26));
  s.push(text(45, 760, "Vitaly.Makhonin@skoltech.ru", 24));
  s.push(text(45, 791, "+7 992 079 07 39", 24));
  s.push(text(930, 690, "Program", 27, { weight: 700 }));
  s.push(text(930, 727, "Data Science", 26));
  s.push(text(1585, 690, "Skoltech supervisor", 27, { weight: 700 }));
  s.push(text(1585, 727, "Alexey Frolov", 26));
  s.push(text(1585, 785, "Company supervisor", 27, { weight: 700 }));
  s.push(text(1585, 822, "Rinat Sultanov", 26));

  // Introduction and objectives.
  s.push(section(40, 930, 1130, 360, "Introduction"));
  s.push(textBlock(62, 990, [
    "A phone camera observes five LEDs as an asynchronous, noisy optical channel.",
    "Hard frame-by-frame decoding was too fragile: a few uncertain observations could",
    "produce a confidently wrong message. The project therefore separates computer",
    "vision from communication decoding. CV produces timestamped soft LED evidence;",
    "a rateless receiver synchronizes the stream and accumulates information online.",
  ], 29, 79, 1.25));
  s.push(rect(62, 1203, 1065, 55, "#EEF1FF", "none", 0, 4));
  s.push(text(82, 1240, "Core novelty: soft rateless reception with persistent online belief propagation.", 26, { weight: 700, fill: C.blue }));

  s.push(section(1200, 930, 1144, 360, "Objectives"));
  s.push(bulletList(1228, 995, [
    "Detect a constrained start-marker / five-LED / end-marker pattern in real time.",
    "Infer symbol timing from a known preamble and sample soft LED observations.",
    "Recover an 8 x 8 binary message without requiring a fixed packet count.",
    "Preserve useful progress through short tracking gaps and reject inconsistent streams.",
    "Integrate the complete receiver workflow into an Android prototype.",
  ], 25, 69, 5));

  // Methods.
  s.push(section(40, 1360, 2304, 970, "Process & Methods"));
  s.push(figureEncoder(62, 1425, 820, 830));
  s.push(figureOptical(905, 1425, 705, 830));
  s.push(figureDecoder(1633, 1425, 689, 830));

  // Results.
  s.push(section(40, 2400, 2304, 600, "Results"));
  s.push(text(65, 2470, "Working end-to-end prototype", 32, { weight: 700 }));
  s.push(bulletList(65, 2520, [
    "Android CameraX + Jetpack Compose receiver with acquisition, decoding progress, cancellation, and final message view.",
    "Browser-based virtual transmitter with the marker pattern, five LED channels, editable binary message, and selectable rates.",
    "Modular interface between CV and protocol: the decoder consumes timestamped soft observations and erasures, not camera frames.",
    "Preamble-based timing estimation, timing-window packet sampling, and online factor-graph decoding are integrated in one workflow.",
    "Rateless reception retains accumulated evidence after short gaps; better CV makes convergence faster without changing the protocol.",
  ], 27, 91, 15));
  s.push(resultPhone(1330, 2450, 975, 500));

  // Conclusions.
  s.push(section(40, 3075, 2304, 225, "Conclusions"));
  s.push(textBlock(62, 3135, [
    "The project established a proprietary architecture for a mobile optical receiver: constrained CV supplies soft evidence, while a", 
    "Raptor-like LDPC+LDGM stream and online BP turn unreliable five-bit observations into accumulated information. Compared with", 
    "hard direct transmission, the receiver is less brittle and does not restart after every short observation gap. The prototype can be", 
    "connected to a future physical LED device; ongoing work focuses on faster mobile vision, hardware validation, and broader phone tests.",
  ], 27, 157, 1.2));
  s.push(text(62, 3282, "Acknowledgements: Alexey Frolov (Skoltech) and Rinat Sultanov (ERKO).", 23, { italic: true, fill: C.muted }));

  s.push(text(42, 3340, "Industrial Immersion", 29, { weight: 700 }));
  s.push(text(WIDTH / 2, 3340, "2026", 29, { weight: 700, anchor: "middle" }));
  s.push(text(WIDTH - 42, 3352, "Skoltech", 48, { weight: 700, anchor: "end" }));
  s.push("</svg>");
  return s.join("\n");
}

const svg = buildPosterSvg();
const svgPath = path.join(DIR, "optical_receiver_poster.svg");
const htmlPath = path.join(DIR, "optical_receiver_poster.html");
const pptxPath = path.join(DIR, "optical_receiver_poster.pptx");
const pdfPath = path.join(DIR, "optical_receiver_poster.pdf");
fs.writeFileSync(svgPath, svg, "utf8");
fs.writeFileSync(
  htmlPath,
  `<!doctype html><html><head><meta charset="utf-8"><style>@page{size:841mm 1189mm;margin:0}html,body{margin:0;width:841mm;height:1189mm;overflow:hidden;background:${C.page}}svg{display:block;width:841mm;height:1189mm}</style></head><body>${svg}</body></html>`,
  "utf8",
);

const pptx = new pptxgen();
pptx.defineLayout({ name: "A0_PORTRAIT", width: 33.11, height: 46.81 });
pptx.layout = "A0_PORTRAIT";
pptx.author = "Vitaly Makhonin";
pptx.company = "ERKO";
pptx.subject = "Skoltech Industrial Immersion 2026";
pptx.title = "Soft Rateless Optical Communication with a Mobile Camera Receiver";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  lang: "en-US",
};
const slide = pptx.addSlide();
slide.background = { color: C.page.slice(1) };
slide.addImage({
  data: `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`,
  x: 0,
  y: 0,
  w: 33.11,
  h: 46.81,
});
await pptx.writeFile({ fileName: pptxPath });

const edgeCandidates = [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
];
const edge = edgeCandidates.find((candidate) => fs.existsSync(candidate));
if (!edge) throw new Error("Microsoft Edge is required to export the poster PDF.");
const uri = new URL(`file:///${htmlPath.replaceAll("\\", "/")}`).href;
const print = spawnSync(
  edge,
  [
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    `--print-to-pdf=${pdfPath}`,
    uri,
  ],
  { encoding: "utf8", timeout: 120000 },
);
if (print.status !== 0 || !fs.existsSync(pdfPath)) {
  throw new Error(`PDF export failed: ${print.stderr || print.stdout}`);
}

console.log(JSON.stringify({ svgPath, htmlPath, pptxPath, pdfPath }, null, 2));
