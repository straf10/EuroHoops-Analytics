// Regenerates public/apple-touch-icon.png (large basket cut) and public/favicon.ico (small cut).
// Geometry mirrors src/components/Logo.astro. Run from web/: node scripts/make-icons.mjs
import { writeFileSync } from "node:fs";
import sharp from "sharp";

const INK = "#111110";
const ORANGE = "#d9590b";
const PAPER = "#f9f9f7";
const BASKET = {
  small: `<path d="M12.2 6h7.6" stroke-width="2.6"/><circle cx="16" cy="9.8" r="2.6" stroke-width="2.2"/>`,
  large: `<path d="M12.6 5.8h6.8" stroke-width="2"/><path d="M16 5.8v1" stroke-width="1.5"/><circle cx="16" cy="9.1" r="2.3" stroke-width="1.6"/>`,
};
const mark = (cut) => `<g transform="translate(0 3)" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <g stroke="${INK}"><path d="M2 3h28" stroke-width="2.6"/>${BASKET[cut]}</g>
  <path d="M4.5 3v9a11.5 11.5 0 0 0 23 0V3" stroke="${ORANGE}" stroke-width="3"/></g>`;
// The mark spans x 2-30 and y 3-17.3+3 within the 32 box; centre it on a paper square with padding.
const render = (size, padFrac, cut) => {
  const s = size * (1 - 2 * padFrac);
  const k = s / 32;
  const off = (size - s) / 2;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}"><rect width="${size}" height="${size}" fill="${PAPER}"/><g transform="translate(${off} ${off}) scale(${k})">${mark(cut)}</g></svg>`;
  return sharp(Buffer.from(svg), { density: 288 }).resize(size, size).png().toBuffer();
};

writeFileSync("public/apple-touch-icon.png", await render(180, 0.14, "large"));

const png = await render(32, 0.04, "small");
const head = Buffer.alloc(22);
head.writeUInt16LE(1, 2);
head.writeUInt16LE(1, 4);
head[6] = 32;
head[7] = 32;
head.writeUInt16LE(1, 10);
head.writeUInt16LE(32, 12);
head.writeUInt32LE(png.length, 14);
head.writeUInt32LE(22, 18);
writeFileSync("public/favicon.ico", Buffer.concat([head, png]));
