// Regenerates public/apple-touch-icon.png and public/favicon.ico from public/favicon.svg.
// Run from web/: node scripts/make-icons.mjs
import { readFileSync, writeFileSync } from "node:fs";
import sharp from "sharp";

const svg = readFileSync("public/favicon.svg", "utf8");
const render = (size, pad, bg) =>
  sharp(Buffer.from(svg), { density: 72 * (size / 24) * 2 })
    .resize(size - pad * 2, size - pad * 2)
    .extend({ top: pad, bottom: pad, left: pad, right: pad, background: bg })
    .png()
    .toBuffer();

writeFileSync("public/apple-touch-icon.png", await render(180, 36, "#f9f9f7"));

// favicon.ico: one 32x32 PNG entry.
const png = await render(32, 2, { r: 249, g: 249, b: 247, alpha: 1 });
const head = Buffer.alloc(22);
head.writeUInt16LE(1, 2); // type: icon
head.writeUInt16LE(1, 4); // one image
head[6] = 32; head[7] = 32;
head.writeUInt16LE(1, 10); // planes
head.writeUInt16LE(32, 12); // bits
head.writeUInt32LE(png.length, 14);
head.writeUInt32LE(22, 18);
writeFileSync("public/favicon.ico", Buffer.concat([head, png]));
