// Environment portals: rectangles (windows) an interior takes its daylight through, for points inside the portal
// region (a room photo's room).
// - Environment NEE draws a share of its samples uniformly over the area of the rectangles in front of the point, one
//   chosen by its rough solid angle, the rest from the map as before; NEE and an escaping BSDF ray both weigh the
//   sample by that mixture's pdf, so the estimate stays unbiased.
// - In a room (info.w), each directional
//   light takes its own shadow ray, traced by roulette where its direction misses every rectangle, and the light RIS
//   draws from a table without them (room), so no candidate is spent on a sun the walls hide.
// - A rough opaque surface in a room aims a share of its continuation rays through the rectangles, at the sunlit
//   ground and sky beyond; its scattering pdf, in NEE's MIS weight too, is then the mixture's.
// - In light RIS, a distant light whose direction misses every rectangle has its target scaled down (never to 0),
//   which only changes which candidate is traced.

/** Rectangles one environment light takes; the uniform holds this many. */
export const MAX_ENV_PORTALS = 16;

/** Bytes of the EnvPortals uniform: four header vec4s, then three vec4s per rectangle. */
export const ENV_PORTALS_UNIFORM_BYTES = (4 + 3 * MAX_ENV_PORTALS) * 16;

export default /* wgsl */ `
struct EnvPortals {
  // x: rectangle count, y: share of environment samples drawn through the rectangles, z: RIS target scale of a
  // distant light whose direction misses every rectangle, w: 1 when a point in the region is a room's (DirectSunInRoom)
  info: vec4f,
  // w: share of a guided room surface's continuations aimed through the rectangles
  regionMin: vec4f,
  regionMax: vec4f,
  // The room's light table (the scene's without directional lights): x its texel offset from the main table's, y its
  // choice count, z its pmf scale (1 / (1 - the directional lights' pmf)), w the chance a sun ray missing every portal
  // is traced
  room: vec4f,
  // Per rectangle: corner.xyz and area, edgeU, edgeV (front: edgeU x edgeV)
  quads: array<vec4f, ${3 * MAX_ENV_PORTALS}>,
}

@group(2) @binding(3) var<uniform> envPortals: EnvPortals;

// The rectangles' total weight at the vertex being shaded when its continuation is guided through them, else 0
var<private> portalGuideTotal: f32 = 0.0;

fn inEnvPortalRegion(p: vec3f) -> bool {
  return HAS_ENV_PORTALS && all(p >= envPortals.regionMin.xyz) && all(p <= envPortals.regionMax.xyz);
}

fn inEnvPortalRoom(p: vec3f) -> bool {
  return envPortals.info.w > 0.5 && inEnvPortalRegion(p);
}

// The chance weight of rectangle i from p: its area over the squared distance to its centre, by the cosine at the
// rectangle (at least 0.05, so a rectangle seen edge-on keeps a share); 0 behind it.
fn envPortalWeight(i: i32, p: vec3f) -> f32 {
  let corner = envPortals.quads[3 * i];
  let front = cross(envPortals.quads[3 * i + 1].xyz, envPortals.quads[3 * i + 2].xyz);
  if (dot(p - corner.xyz, front) <= 0.0) {
    return 0.0;
  }
  let toPoint = p - (corner.xyz + 0.5 * (envPortals.quads[3 * i + 1].xyz + envPortals.quads[3 * i + 2].xyz));
  let distanceSquared = max(dot(toPoint, toPoint), 1e-8);
  let cosine = dot(toPoint, front) * inverseSqrt(distanceSquared) / max(corner.w, 1e-12);
  return corner.w * max(cosine, 0.05) / max(distanceSquared, 0.25 * corner.w);
}

fn envPortalTotalWeight(p: vec3f) -> f32 {
  var total = 0.0;
  for (var i = 0; i < i32(envPortals.info.x); i++) {
    total += envPortalWeight(i, p);
  }
  return total;
}

// Solid-angle pdf of the rectangle strategy at p for direction dir (unit): the chance of each rectangle the ray
// leaves through, by distance squared over its projected area.
fn envPortalsPdf(p: vec3f, dir: vec3f, total: f32) -> f32 {
  var pdf = 0.0;
  for (var i = 0; i < i32(envPortals.info.x); i++) {
    let weight = envPortalWeight(i, p);
    if (weight <= 0.0) {
      continue;
    }
    let corner = envPortals.quads[3 * i].xyz;
    let edgeU = envPortals.quads[3 * i + 1].xyz;
    let edgeV = envPortals.quads[3 * i + 2].xyz;
    let front = cross(edgeU, edgeV);
    let facing = dot(dir, front);
    if (facing >= 0.0) {
      continue;
    }
    let t = dot(corner - p, front) / facing;
    if (t <= 0.0) {
      continue;
    }
    let local = p + t * dir - corner;
    let a = dot(local, edgeU) / dot(edgeU, edgeU);
    let b = dot(local, edgeV) / dot(edgeV, edgeV);
    if (a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0) {
      pdf += weight / total * t * t / -facing;
    }
  }
  return pdf;
}

// Whether dir from p leaves through a rectangle in front of p.
fn envPortalsPassed(p: vec3f, dir: vec3f) -> bool {
  for (var i = 0; i < i32(envPortals.info.x); i++) {
    let corner = envPortals.quads[3 * i].xyz;
    let edgeU = envPortals.quads[3 * i + 1].xyz;
    let edgeV = envPortals.quads[3 * i + 2].xyz;
    let front = cross(edgeU, edgeV);
    let facing = dot(dir, front);
    let height = dot(p - corner, front);
    if (facing >= 0.0 || height <= 0.0) {
      continue;
    }
    let local = p + (height / -facing) * dir - corner;
    let a = dot(local, edgeU) / dot(edgeU, edgeU);
    let b = dot(local, edgeV) / dot(edgeV, edgeV);
    if (a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0) {
      return true;
    }
  }
  return false;
}

// The RIS target scale of a distant light's direction from p: 1 through a rectangle (or outside the region), else
// the uniform's small share.
fn envPortalsTargetScale(p: vec3f, dir: vec3f) -> f32 {
  if (!inEnvPortalRegion(p) || envPortalsPassed(p, dir)) {
    return 1.0;
  }
  return envPortals.info.z;
}

// An environment sample from p: through the rectangles (share info.y) or from the map. Outside the region, or with
// no rectangle in front of p, the map's sample as sampleEnvLight draws it.
fn sampleEnvLightFrom(p: vec3f, randomV2: vec2f, lightSample: ptr<function, LightSampleRec>) {
  if (!inEnvPortalRegion(p)) {
    sampleEnvLight(randomV2, lightSample);
    return;
  }
  let total = envPortalTotalWeight(p);
  if (total <= 0.0) {
    sampleEnvLight(randomV2, lightSample);
    return;
  }
  let share = envPortals.info.y;
  var dir: vec3f;
  if (randomV2.x < share) {
    dir = sampleEnvPortalDirection(p, vec2f(randomV2.x / share, randomV2.y), total);
  } else {
    var mapUv: vec2f;
    var mapPdf: f32;
    dir = sampleEnvmap(vec2f((randomV2.x - share) / max(1.0 - share, 1e-6), randomV2.y), &mapUv, &mapPdf);
  }
  let uv = cartesianToEquirect(dir);
  (*lightSample).direction = dir;
  (*lightSample).emission = textureLinear(envMap, uv).rgb;
  (*lightSample).dist = INF;
  (*lightSample).pdf = share * envPortalsPdf(p, dir, total) + (1.0 - share) * envMapPdf(uv);
}

// A direction from p through the rectangles: one by its weight (random.x, rescaled), then a point uniformly over its
// area. Its pdf is envPortalsPdf's.
fn sampleEnvPortalDirection(p: vec3f, random: vec2f, total: f32) -> vec3f {
  var pick = random.x * total;
  var chosen = -1;
  var last = 0;
  for (var i = 0; i < i32(envPortals.info.x); i++) {
    let weight = envPortalWeight(i, p);
    if (weight <= 0.0) {
      continue;
    }
    if (pick < weight) {
      chosen = i;
      pick = pick / weight;
      break;
    }
    pick -= weight;
    last = i;
  }
  if (chosen < 0) {
    // Rounding past the last weight: the last rectangle in front of p
    chosen = last;
    pick = 1.0;
  }
  let point = envPortals.quads[3 * chosen].xyz
    + clamp(pick, 0.0, 1.0) * envPortals.quads[3 * chosen + 1].xyz
    + random.y * envPortals.quads[3 * chosen + 2].xyz;
  return normalize(point - p);
}

// The environment strategy's solid-angle pdf at p for dir, as sampleEnvLightFrom draws it.
fn envLightPdfFrom(p: vec3f, dir: vec3f, uv: vec2f) -> f32 {
  if (inEnvPortalRegion(p)) {
    let total = envPortalTotalWeight(p);
    if (total > 0.0) {
      let share = envPortals.info.y;
      return share * envPortalsPdf(p, dir, total) + (1.0 - share) * envMapPdf(uv);
    }
  }
  return envMapPdf(uv);
}

// The NEE pdf of the environment at p for dir, its selection included: from the room's table in a room
// (scaled pmf), from the light table elsewhere.
fn envNeePdfFrom(p: vec3f, dir: vec3f, uv: vec2f) -> f32 {
  return envLightPdfFrom(p, dir, uv) * rtUniform.envLightPmf * select(1.0, envPortals.room.z, inEnvPortalRoom(p));
}

// The rectangles' total weight at p when a surface of this material there guides a share (regionMin.w) of its
// continuations through them: rough, opaque, single-layer surfaces in a room (other lobes keep their own sampling).
fn envPortalGuideTotal(p: vec3f, ma: OpenPbrResolvedMaterial) -> f32 {
  if (!inEnvPortalRoom(p) || envPortals.regionMin.w <= 0.0 || ma.transmissionWeight > 0.0 || ma.subsurfaceWeight > 0.0 ||
      ma.coatWeight > 0.0 || ma.thinWalled || ma.specularRoughness < 0.3) {
    return 0.0;
  }
  return envPortalTotalWeight(p);
}

// A guided vertex's scattering pdf for dir: the share through the rectangles mixed with the BSDF's own pdf.
fn envPortalGuidedPdf(p: vec3f, dir: vec3f, bsdfPdf: f32) -> f32 {
  let share = envPortals.regionMin.w;
  return share * envPortalsPdf(p, dir, portalGuideTotal) + (1.0 - share) * bsdfPdf;
}
`;
