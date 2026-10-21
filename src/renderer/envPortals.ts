import type { EnvironmentPortals } from '../scene/EnvironmentLight.js';
import { ENV_PORTALS_UNIFORM_BYTES, MAX_ENV_PORTALS } from './wgsl/chunks/envPortals.wgsl.js';

export { ENV_PORTALS_UNIFORM_BYTES, MAX_ENV_PORTALS };

/**
 * Share of a portal region's environment samples drawn through its rectangles; the rest come from the map, so
 * light that reaches the region another way (through a door from another room's window) is still sampled.
 */
export const ENV_PORTAL_SHARE = 0.8;

/**
 * Light RIS target scale of a distant light whose direction from a point in the region misses every rectangle:
 * small, so candidates through a window win, never 0, so the estimate keeps light that arrives another way.
 */
export const ENV_PORTAL_MISS_SCALE = 0.05;

/**
 * Chance that a room point traces its shadow ray to a directional light whose direction misses every rectangle (the
 * walls hide it there all but always); a traced one counts 1 / this, so the estimate stays unbiased.
 */
export const ENV_PORTAL_SUN_ROULETTE = 0.125;

/** Share of a rough opaque room surface's continuation rays aimed through the rectangles, at what lies beyond */
export const ENV_PORTAL_GUIDE_SHARE = 0.25;

/**
 * Packs portals into the EnvPortals uniform layout (header: count, share, miss scale, room flag; region min; region
 * max; room table; then per rectangle corner + area, edgeU, edgeV). Degenerate rectangles are skipped and at most
 * MAX_ENV_PORTALS kept. `room`: the light table a point in the region draws from, the scene's without its
 * directional lights (each of those takes its own shadow ray there): its texel offset from the main table, its choice
 * count and its pmf scale. Returns the rectangle count; 0 leaves the shader's portal code switched off.
 */
export function packEnvPortals(
  target: Float32Array,
  portals: EnvironmentPortals | undefined,
  room?: { tableOffset: number; choiceCount: number; pmfScale: number }
): number {
  if (target.length * 4 < ENV_PORTALS_UNIFORM_BYTES) throw new RangeError('EnvPortals buffer too small');
  target.fill(0);
  if (!portals) return 0;
  let count = 0;
  for (const { corner, edgeU, edgeV } of portals.rectangles) {
    if (count === MAX_ENV_PORTALS) {
      console.warn(`Too many environment portals, only the first ${MAX_ENV_PORTALS} are used.`);
      break;
    }
    const nx = edgeU[1] * edgeV[2] - edgeU[2] * edgeV[1];
    const ny = edgeU[2] * edgeV[0] - edgeU[0] * edgeV[2];
    const nz = edgeU[0] * edgeV[1] - edgeU[1] * edgeV[0];
    const area = Math.hypot(nx, ny, nz);
    if (!(area > 1e-8) || ![...corner, ...edgeU, ...edgeV].every(Number.isFinite)) continue;
    const offset = 16 + count * 12;
    target.set([...corner, area, ...edgeU, 0, ...edgeV, 0], offset);
    count++;
  }
  if (count === 0) return 0;
  target.set([count, ENV_PORTAL_SHARE, ENV_PORTAL_MISS_SCALE, room ? 1 : 0], 0);
  target.set([...portals.region.min, ENV_PORTAL_GUIDE_SHARE, ...portals.region.max, 0], 4);
  if (room) target.set([room.tableOffset, room.choiceCount, room.pmfScale, ENV_PORTAL_SUN_ROULETTE], 12);
  return count;
}
