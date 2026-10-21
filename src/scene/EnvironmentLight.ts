import Texture from './Texture.js';

/**
 * A rectangle the environment lights an interior through, such as a window: `corner` plus the two edges `edgeU`
 * and `edgeV`, in world space. Its front, `edgeU × edgeV`, faces the interior.
 */
export interface EnvironmentPortal {
  /** One corner of the rectangle; no default. */
  corner: [number, number, number];
  /** First edge from the corner; no default. */
  edgeU: [number, number, number];
  /** Second edge from the corner; no default. */
  edgeV: [number, number, number];
}

/**
 * Portals of an interior lit through its windows, such as a room photographed from inside. Points inside `region`
 * aim most environment samples through the rectangles in front of them; each directional light (the sun) takes a
 * shadow ray of its own there, rarely where its direction misses every rectangle, instead of competing with the sky
 * for light candidates; and rough opaque surfaces aim a share of their bounces through the rectangles, at the sunlit
 * ground and sky outside. All of it stays unbiased: light that reaches the region another way (through a door from
 * another room's window) is still found, only less often. WebGPU only; WebGL samples the map as usual.
 */
export interface EnvironmentPortals {
  /** At most 16 rectangles; more are ignored. No default. */
  rectangles: EnvironmentPortal[];
  /**
   * World-space box of the interior the rectangles light: `corner` plus the three edges from it, so a room of a
   * rotated building keeps its own box and the terrace outside stays out. Any parallelepiped works; edges that span
   * no volume switch the portals off. No default.
   */
  region: {
    corner: [number, number, number];
    edges: [[number, number, number], [number, number, number], [number, number, number]];
  };
}

/**
 * Image-based lighting from an equirectangular (latitude-longitude) HDR map,
 * for example one loaded with `loadRGBE()`. Importance sampled.
 */
export default class EnvironmentLight {
  /** Light discriminator; always `EnvironmentLight`. */
  readonly type = 'EnvironmentLight' as const;
  /** Equirectangular radiance map in linear color. */
  map: Texture;
  /** Interior portals the environment is sampled through. Default undefined (none), and then no own property. */
  declare portals?: EnvironmentPortals;
  constructor(params: {
    /** Required equirectangular radiance map; no default. */ map: Texture;
    /** Interior portals the environment is sampled through. Default undefined (none). */ portals?: EnvironmentPortals;
  }) {
    this.map = params.map;
    if (params.portals) this.portals = params.portals;
  }
}
