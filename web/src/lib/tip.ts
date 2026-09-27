// Tooltip bodies. Base.astro renders data-tip as HTML, so every tooltip is built here: the
// template's own text may carry <b> and <br>, and every interpolated value is escaped.

import { esc } from "./format";

/** tip`<b>${name}</b><br>${about}`: the markup is the template's, every value is text. */
export function tip(strings: TemplateStringsArray, ...values: unknown[]): string {
  return strings.reduce((html, part, i) => html + part + (i < values.length ? esc(String(values[i])) : ""), "");
}

/** A tooltip body as a data-tip="..." attribute (with its leading space) for string-built markup.
 * Only & and " need escaping inside one, so the value the browser reads back is the body exactly
 * and <b> and <br> stay as written, as Astro writes data-tip={...} in components. */
export const dataTip = (html: string): string =>
  ` data-tip="${html.replace(/[&"]/g, (c) => (c === "&" ? "&amp;" : "&quot;"))}"`;
