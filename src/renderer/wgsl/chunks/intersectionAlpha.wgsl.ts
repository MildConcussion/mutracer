import { mapAlphaUsageConstants } from '../../materialConstants.js';
import { uvProjectionConstants } from '../../materialConstants.js';
import {
  materialAlphaModeConstants
} from '../../materialConstants.js';

/**
 * Material coverage shared by software BVH traversal and native RayQuery
 * candidate filtering. The caller supplies local-space hit data so UV
 * projection and stochastic glTF BLEND semantics stay backend-independent.
 */
export function createIntersectionAlphaWgsl(hasAnisotropyDirections = true) {
  return /* wgsl */ `
fn intersectionAlphaPasses(
  initialMaterialIndex: i32,
  index: vec3i,
  hitBarycentric: vec3f,
  hitPosition: vec3f,
  hitNormal: vec3f
) -> bool {
  var materialIndex = initialMaterialIndex;
  let m0 = vec3i(attrBuffers[index.x * ATTR_STRIDE + MATERIALINDEX_OFF]${hasAnisotropyDirections ? '.xyz' : ''});
  let m1 = vec3i(attrBuffers[index.y * ATTR_STRIDE + MATERIALINDEX_OFF]${hasAnisotropyDirections ? '.xyz' : ''});
  let m2 = vec3i(attrBuffers[index.z * ATTR_STRIDE + MATERIALINDEX_OFF]${hasAnisotropyDirections ? '.xyz' : ''});

  if (m0.x > 0 || m1.x > 0 || m2.x > 0) {
    materialIndex = randomPickMaterial(m0, m1, m2, hitBarycentric);
  }

  let ma = materialsBuffer[materialIndex];
  var alpha = ma.alpha;
  if (USE_DIFFUSE_MAPS &&
      ma.mapAlphaUsage == ${mapAlphaUsageConstants.alpha}) {
    let diffuseMapIndex = i32(ma.diffuseMapIndex);
    if (diffuseMapIndex >= 0) {
      let uvProjection = i32(ma.uvProjection);
      let uv0 = attrBuffers[index.x * ATTR_STRIDE + UV_OFF].xy;
      let uv1 = attrBuffers[index.y * ATTR_STRIDE + UV_OFF].xy;
      let uv2 = attrBuffers[index.z * ATTR_STRIDE + UV_OFF].xy;
      var uv = hitBarycentric.x * uv0 +
        hitBarycentric.y * uv1 + hitBarycentric.z * uv2;
      if (uvProjection > 0) {
        uv = projectUV(uvProjection, hitPosition, hitNormal);
      }
      uv *= ma.uvScale;
      if (uvProjection == ${uvProjectionConstants.triplanar}) {
        uv = fract(uv);
      }
      alpha *= sampleMap(diffuseMapIndex, uv).a;
    }
  }

  if (USE_ALPHA_MAPS) {
    let alphaMapIndex = i32(ma.alphaMapIndex);
    if (alphaMapIndex >= 0) {
      let uvProjection = i32(ma.uvProjection);
      let uv0 = attrBuffers[index.x * ATTR_STRIDE + UV_OFF].xy;
      let uv1 = attrBuffers[index.y * ATTR_STRIDE + UV_OFF].xy;
      let uv2 = attrBuffers[index.z * ATTR_STRIDE + UV_OFF].xy;
      var uv = hitBarycentric.x * uv0 +
        hitBarycentric.y * uv1 + hitBarycentric.z * uv2;
      if (uvProjection > 0) {
        uv = projectUV(uvProjection, hitPosition, hitNormal);
      }
      uv *= ma.uvScale;
      if (uvProjection == ${uvProjectionConstants.triplanar}) {
        uv = fract(uv);
      }
      alpha *= sampleMap(alphaMapIndex, uv).g;
    }
  }

  let alphaMode = i32(ma.alphaMode);
  if (alphaMode == ${materialAlphaModeConstants.opaque}) {
    alpha = 1.0;
  }
  else if (alphaMode == ${materialAlphaModeConstants.mask}) {
    alpha = select(0.0, 1.0, alpha >= ma.alphaCutoff);
  }
  if (alphaMode == ${materialAlphaModeConstants.blend}) {
    return hashedSample(1u) <= alpha;
  }
  return alpha >= 1.0;
}

// Closest-hit traversal skips the cut-out texels of alpha-MASK materials in
// place, so a ray crosses masked cards (foliage) in one traversal instead of
// returning a hit the path then skips and restarting from the root. True
// only where getMaterial would resolve this hit to zero coverage: a mask on
// the instance material, read through the first UV set without projection.
// Every other case is left to the path's own coverage test. Draws no random
// numbers, so the sample streams are unchanged.
fn intersectionMaskRejects(
  materialIndex: i32,
  index: vec3i,
  hitBarycentric: vec3f
) -> bool {
  let ma = materialsBuffer[materialIndex];
  if (i32(ma.alphaMode) != ${materialAlphaModeConstants.mask} ||
      i32(ma.uvProjection) > 0) {
    return false;
  }
  let m0 = vec3i(attrBuffers[index.x * ATTR_STRIDE + MATERIALINDEX_OFF]${hasAnisotropyDirections ? '.xyz' : ''});
  let m1 = vec3i(attrBuffers[index.y * ATTR_STRIDE + MATERIALINDEX_OFF]${hasAnisotropyDirections ? '.xyz' : ''});
  let m2 = vec3i(attrBuffers[index.z * ATTR_STRIDE + MATERIALINDEX_OFF]${hasAnisotropyDirections ? '.xyz' : ''});
  if (m0.x > 0 || m1.x > 0 || m2.x > 0) {
    return false;
  }

  var uv = vec2f(0.0);
  if (USE_UV) {
    uv = hitBarycentric.x * attrBuffers[index.x * ATTR_STRIDE + UV_OFF].xy +
      hitBarycentric.y * attrBuffers[index.y * ATTR_STRIDE + UV_OFF].xy +
      hitBarycentric.z * attrBuffers[index.z * ATTR_STRIDE + UV_OFF].xy;
  }
  uv *= ma.uvScale;

  var alpha = ma.alpha;
  if (USE_DIFFUSE_MAPS &&
      ma.mapAlphaUsage == ${mapAlphaUsageConstants.alpha}) {
    let diffuseMapIndex = i32(ma.diffuseMapIndex);
    if (diffuseMapIndex >= 0) {
      if (mapInfo[diffuseMapIndex + NUM_MAPS * 2].x > 0.5) { return false; }
      alpha *= sampleMap(diffuseMapIndex, uv).a;
    }
  }
  if (USE_ALPHA_MAPS) {
    let alphaMapIndex = i32(ma.alphaMapIndex);
    if (alphaMapIndex >= 0) {
      if (mapInfo[alphaMapIndex + NUM_MAPS * 2].x > 0.5) { return false; }
      alpha *= sampleMap(alphaMapIndex, uv).g;
    }
  }
  return alpha < ma.alphaCutoff;
}
`;


}
export default createIntersectionAlphaWgsl();
