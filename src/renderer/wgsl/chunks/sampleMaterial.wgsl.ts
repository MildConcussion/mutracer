// Next-event estimation with resampled importance sampling over the light
// candidates, followed by BSDF (or phase function) sampling, as described in
// http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Direct_Lighting.html#EstimatingtheDirectLightingIntegral
export function createSampleMaterialWgsl(hasAnisotropyDirections = true, hasCoatNormals = false) {
  return /* wgsl */ `

// A thin pane (MaterialTable isThinPane) as its smooth thin-wall OpenPBR lobes
// give it: a mirror of reflectance specularColor * Rw and a straight-through
// transmission transmissionColor^(1 / cos(theta_t)) * (1 - Rw), where
// Rw = 2F / (1 + F) is the window reflectance of one interface's Fresnel F.
fn thinPaneEta(ior: f32, specularWeight: f32) -> f32 {
  return openpbr_apply_specular_weight_to_ior(ior, max(specularWeight, 0.0));
}

fn thinPaneReflectance(eta: f32, cosTheta: f32) -> f32 {
  let fresnel = openpbr_dielectric_fresnel(clamp(abs(cosTheta), 0.0, 1.0), 1.0 / eta);
  return 2.0 * fresnel / (1.0 + fresnel);
}

fn thinPaneTransmission(eta: f32, transmissionColor: vec3f, cosTheta: f32) -> vec3f {
  let cosine = clamp(abs(cosTheta), 0.0, 1.0);
  let transmittedSineSquared = (1.0 - cosine * cosine) / max(eta * eta, 1e-9);
  if (transmittedSineSquared >= 1.0) {
    return vec3f(0.0);
  }
  let tint = pow(
    max(transmissionColor, vec3f(0.0)),
    vec3f(1.0 / max(sqrt(1.0 - transmittedSineSquared), 1e-5))
  );
  return tint * (1.0 - thinPaneReflectance(eta, cosine));
}

// A path meets a thin pane: it passes straight through (true) or mirrors off
// it (false), chosen by the luminance of the two responses. Passing is no
// scattering event: the ray keeps its direction, BSDF pdf and bounce, and the
// last vertex is kept for the MIS of a light reached behind the pane.
fn crossThinPane(
  si: ptr<function, SurfaceInteraction>,
  ma: ptr<function, OpenPbrResolvedMaterial>,
  path: ptr<function, Path>
) -> bool {
  let incident = (*path).ray.d;
  let cosTheta = dot((*si).faceNormal, incident);
  let eta = thinPaneEta((*ma).specularIor, (*ma).specularWeight);
  let transmission = thinPaneTransmission(eta, (*ma).transmissionColor, cosTheta);
  let reflection = (*ma).specularColor * thinPaneReflectance(eta, cosTheta);
  let passWeight = dot(transmission, luminance);
  let total = passWeight + dot(reflection, luminance);
  if (!(total > 0.0)) {
    (*path).abort = true;
    return false;
  }
  let passProbability = passWeight / total;
  if (hashedSample(5u) < passProbability) {
    (*path).beta *= transmission / passProbability;
    if (!(*path).paneCrossed) {
      (*path).paneOrigin = (*path).ray.o;
      (*path).paneCrossed = true;
    }
    (*path).paneEvents += 1;
    (*path).ray.o = (*si).position + incident * EPS;
    return true;
  }
  // A smooth mirror: a delta event no light sample can reach, so whatever its
  // reflection meets gets the full MIS weight.
  let facing = select((*si).faceNormal, -(*si).faceNormal, cosTheta > 0.0);
  (*path).beta *= reflection / (1.0 - passProbability);
  (*path).ray.d = reflect(incident, facing);
  (*path).ray.o = (*si).position + facing * EPS;
  (*path).scatteringPdf = THIN_PANE_DELTA_PDF;
  (*path).scatteringIsDelta = false;
  (*path).isNextRefraction = false;
  (*path).fromEye = false;
  (*path).paneCrossed = false;
  return false;
}

struct ShadowRayWork {
  ray: Ray,
  maxDistance: f32,
  triangleIndex: i32,
  contribution: vec3f,
  // RIS target of the candidate. The discrete light-selection PMF is excluded
  // and represented by the candidate's unbiased contribution weight instead.
  risTarget: f32,
  lightSelectionPmf: f32,
  valid: bool,
}

fn PrepareDirectLight(
  si: ptr<function, SurfaceInteraction>,
  ma: OpenPbrResolvedMaterial,
  prepared: OpenPbrPreparedBsdf,
  path: ptr<function, Path>,
  ffnormal: vec3f,
  isSurface: bool,
  candidateIndex: u32,
  candidateCount: u32,
  room: bool,
  shadowWork: ptr<function, ShadowRayWork>
) -> bool {
  (*shadowWork).valid = false;
  (*shadowWork).lightSelectionPmf = 0.0;
  (*shadowWork).risTarget = 0.0;
  let lightRnd = vec3f(sample2D(lightDim(candidateIndex, 1u)), sample1D(lightDim(candidateIndex, 0u)));

  var lightSample: LightSampleRec;
  if (room) {
    sampleLightChoiceTable(
      (*si).position,
      stratifyLightChoiceRandom(lightRnd, candidateIndex, candidateCount),
      i32(envPortals.room.x),
      i32(envPortals.room.y),
      envPortals.room.z,
      &lightSample
    );
  } else {
    sampleLightForNEECandidate(
      (*si).position,
      lightRnd,
      candidateIndex,
      candidateCount,
      &lightSample
    );
  }
  return PrepareLightSample(si, ma, prepared, path, ffnormal, isSurface, lightSample, shadowWork);
}

// The shadow-ray work of one drawn light sample: its unshadowed contribution with the NEE MIS weight, and the RIS
// target. False when it contributes nothing.
fn PrepareLightSample(
  si: ptr<function, SurfaceInteraction>,
  ma: OpenPbrResolvedMaterial,
  prepared: OpenPbrPreparedBsdf,
  path: ptr<function, Path>,
  ffnormal: vec3f,
  isSurface: bool,
  drawn: LightSampleRec,
  shadowWork: ptr<function, ShadowRayWork>
) -> bool {
  (*shadowWork).valid = false;
  (*shadowWork).lightSelectionPmf = 0.0;
  (*shadowWork).risTarget = 0.0;
  var lightSample = drawn;
  var Li = lightSample.emission;

  let lightDir = lightSample.direction;
  let viewDir = -(*path).ray.d;
  let lightPdf = lightSample.pdf;
  let lightSelectionPmf = lightSample.selectionPmf;
  // Keep direct-light culling consistent with the two-sided BSDF evaluation.
  // The raw imported normal can point away from the camera on interior walls.
  ${hasCoatNormals ? /* wgsl */ `var cosThetaL = dot(ffnormal, lightDir);
  if (isSurface && ma.coatNormalEnabled) { cosThetaL = dot(ma.geometricNormal, lightDir); }` : 'let cosThetaL = dot(ffnormal, lightDir);'}

  if (lightPdf <= 0.0 || lightSelectionPmf <= 0.0 ||
      max(max(Li.x, Li.y), Li.z) <= 0.0) {
    return false;
  }

  var scatteringPdf = 0.0;
  var f = vec3f(0.0);
  if (isSurface) {
    // Evaluate the cheap BSDF terms before paying for a scene traversal.
    // Back-facing, delta, and otherwise zero-contribution samples can stop here.
    if (cosThetaL <= 0.0 && ma.transmissionWeight <= 0.0 && ma.subsurfaceWeight <= 0.0) {
      return false;
    }
    f = openpbr_eval_prepared(prepared, lightDir, &scatteringPdf);
  }
  else {
    scatteringPdf = sampling_henyey_greenstein_phase(
      dot(viewDir, lightDir),
      (*si).mediumAnisotropy
    );
    f = vec3f(scatteringPdf);
  }

  if (scatteringPdf <= 0.0 || max(max(f.x, f.y), f.z) <= 0.0) {
    return false;
  }

  var lightRay: Ray;
  var shadowOrigin = (*si).position + EPS * lightDir;
  if (isSurface) {
    shadowOrigin = offsetSurfaceRay((*si).position, (*si).faceNormal, lightDir);
    let shadowTerminatorCutoff = 0.1;
    var normalLight = dot((*si).normal, lightDir);
${hasCoatNormals ? /* wgsl */ `    if (ma.coatNormalEnabled) { normalLight = dot((*si).faceNormal, lightDir); }` : ''}
    let geometricNormal = select(
      (*si).faceNormal,
      -(*si).faceNormal,
      normalLight < 0.0
    );
    normalLight = abs(normalLight);
    let geometricLight = dot(geometricNormal, lightDir);
    let offsetAmount = select(
      clamp(1.0 - geometricLight / shadowTerminatorCutoff, 0.0, 1.0),
      clamp(
        2.0 - (geometricLight + normalLight) / shadowTerminatorCutoff,
        0.0,
        1.0
      ),
      normalLight < shadowTerminatorCutoff
    );
    shadowOrigin += (*si).shadowOffset * offsetAmount;
  }
  // Falcor offsets sampled emissive positions along the light normal, then
  // derives the visibility ray from that offset endpoint. Without the target
  // offset, our any-hit traversal sees the sampled light triangle itself and
  // rejects virtually every emissive NEE sample as occluded.
  var visibilityDirection = lightDir;
  var visibilityDistance = lightSample.dist - EPS;
  if (lightSample.sourceType == 1u) {
    let lightEndpoint =
      (*si).position + lightSample.direction * lightSample.dist;
    let offsetLightEndpoint = lightEndpoint + EPS * lightSample.normal;
    let toOffsetLight = offsetLightEndpoint - shadowOrigin;
    visibilityDistance = length(toOffsetLight);
    visibilityDirection = normalize(toOffsetLight);
  }
  initRay(&lightRay, shadowOrigin, visibilityDirection);

  // The continuation of a surface transmission sees the displayed background.
  // Evaluate that same endpoint for the environment proposal, so both MIS
  // strategies estimate the same radiance even for a separate/grounded skybox.
  if (isSurface && cosThetaL < 0.0 && lightSample.sourceType == 0u) {
    let background = sampleSceneBackground(lightRay);
    Li = background.rgb * background.a;
  }
  if (SPECTRAL) {
    Li = sampling_spectral_rgb_filter(Li, (*si).wavelength);
  }

  // A directional source is a delta distribution and cannot be reached by
  // the BSDF-sampling strategy, so its light-sampled contribution has weight 1.
  var misScatteringPdf = scatteringPdf;
  if (HAS_ENV_PORTALS && isSurface && portalGuideTotal > 0.0) {
    misScatteringPdf = envPortalGuidedPdf((*si).position, lightDir, scatteringPdf);
  }
  let misWeight = select(
    neeMISWeight(lightPdf, misScatteringPdf),
    1.0,
    lightSample.isDelta
  );
  if (misWeight > 0.0) {
    let unweightedContribution = Li * f / lightPdf * (*path).beta;
    (*shadowWork).risTarget = max(
      dot(max(unweightedContribution * lightSelectionPmf, vec3f(0.0)), luminance),
      0.0
    );
    if (HAS_ENV_PORTALS && lightSample.dist >= INF) {
      (*shadowWork).risTarget *= envPortalsTargetScale((*si).position, lightDir);
    }
    (*shadowWork).ray = lightRay;
    (*shadowWork).maxDistance = visibilityDistance;
    (*shadowWork).triangleIndex = (*si).triangleIndex;
    (*shadowWork).contribution = misWeight * unweightedContribution;
    (*shadowWork).lightSelectionPmf = lightSelectionPmf;
    (*shadowWork).valid = true;
    return true;
  }
  return false;
}

fn DirectLight(
  si: ptr<function, SurfaceInteraction>,
  ma: OpenPbrResolvedMaterial,
  prepared: OpenPbrPreparedBsdf,
  path: ptr<function, Path>,
  ffnormal: vec3f,
  isSurface: bool
) {
  // In a room photo's room the sun takes its own shadow ray and the candidates come from the room's light table,
  // which leaves it out (DirectSunInRoom)
  let room = HAS_ENV_PORTALS && inEnvPortalRoom((*si).position);
  if (room) {
    DirectSunInRoom(si, ma, prepared, path, ffnormal, isSurface);
  }
  let candidateCount = neeCandidateCount();
  var selected: ShadowRayWork;
  selected.valid = false;
  var selectedTarget = 0.0;
  var scalarWeightSum = 0.0;
  // Streaming weighted reservoir sampling over the light candidates. Only the
  // selected candidate traces a shadow ray.
  for (var candidateIndex = 0u; candidateIndex < candidateCount; candidateIndex++) {
    var candidate: ShadowRayWork;
    let candidateValid = PrepareDirectLight(
      si, ma, prepared, path, ffnormal, isSurface,
      candidateIndex, candidateCount, room, &candidate);
    if (candidateCount == 1u) {
      if (!candidateValid) { return; }
      selected = candidate;
      break;
    }
    if (!candidateValid) { continue; }
    let scalarWeight = candidate.risTarget / max(candidate.lightSelectionPmf, 1e-20);
    if (!(scalarWeight > 0.0) || scalarWeight != scalarWeight) { continue; }
    scalarWeightSum += scalarWeight;
    if (!selected.valid ||
        lightCandidateRandom(candidateIndex) * scalarWeightSum <= scalarWeight) {
      selected = candidate;
      selectedTarget = candidate.risTarget;
    }
  }
  var candidateUCW = 1.0;
  if (candidateCount > 1u) {
    if (!selected.valid || selectedTarget <= 0.0) { return; }
    // RIS unbiased contribution weight, normalized by the target rather than
    // target / selection PMF.
    candidateUCW = selected.lightSelectionPmf * scalarWeightSum /
      (f32(candidateCount) * selectedTarget);
  }
  if (selected.valid &&
      !intersectSceneShadow(selected.ray, selected.maxDistance, selected.triangleIndex)) {
    // Thin panes the shadow ray crossed filter the light (1 without panes).
    (*path).li += selected.contribution * candidateUCW * shadowPaneTransmittance;
  }
}

fn emissiveHitMisWeight(
  si: ptr<function, SurfaceInteraction>,
  path: ptr<function, Path>
) -> f32 {
  if ((*path).fromEye || rtUniform.lightWeightTotal <= 0.0 ||
      i32(materialsBuffer[(*si).materialIndex].matType) == UNLIT) {
    return 1.0;
  }
  let samplingLuminance = dot(
    materialsBuffer[(*si).materialIndex].emission,
    luminance
  );
  let cosLight = dot((*si).faceNormal, -(*path).ray.d);
  if (samplingLuminance <= 0.0 || cosLight <= 0.0) {
    return 1.0;
  }
  var origin = (*path).ray.o;
  if (HAS_THIN_PANE && (*path).paneCrossed) {
    origin = (*path).paneOrigin;
  }
  let hitDistance = distance((*si).position, origin);
  var lightPdf = samplingLuminance / rtUniform.lightWeightTotal *
    hitDistance * hitDistance / max(cosLight, 1e-8);
  if (HAS_ENV_PORTALS && inEnvPortalRoom(origin)) {
    lightPdf *= envPortals.room.z;
  }
  return bsdfMISWeight((*path).scatteringPdf, lightPdf);
}

${hasCoatNormals ? /* wgsl */ `
// Keep each shading frame in the geometric/view hemisphere. This correction
// does not replace the face normal used for topology or ray offsets.
fn coat_frame_normal(input: vec3f, face: vec3f, view: vec3f) -> vec3f {
  let front = dot(face, view) >= 0.0;
  let ffFace = select(-face, face, front);
  let ffInput = select(-input, input, front);
  if (dot(input, face) <= 0.0 || dot(ffInput, view) <= 1e-6) { return ffFace; }
  return ffInput;
}
` : ''}
fn SampleMaterial(
  si: ptr<function, SurfaceInteraction>,
  ma: ptr<function, OpenPbrResolvedMaterial>,
  path: ptr<function, Path>
) {
${hasCoatNormals ? /* wgsl */ `
  let originalFace = (*si).faceNormal;
  let originalNormal = (*si).normal;
` : ''}
  var ndv1: f32 = -dot((*si).faceNormal, (*path).ray.d);
  var ndv2: f32 = -dot((*si).normal, (*path).ray.d);
  if (((*ma).transmissionWeight > 0.0 || (*ma).subsurfaceWeight > 0.0) && (*ma).thinWalled) {
    // Always treat thin material as entering medium
    if (ndv1 < 0.0) {
      (*si).faceNormal = -(*si).faceNormal;
      ndv1 = -ndv1;
    }
    if (ndv2 < 0.0) {
      (*si).normal = -(*si).normal;
      ndv2 = -ndv2;
    }
  }
  // In same edge cases normal is not torwards the camera while we can still see the face.
  // We need to choose faceNormal in this case.
  var useFaceN: bool = ndv1 >= 0.0 && ndv2 < 0.0;
  var ndv: f32 = select(ndv2, ndv1, useFaceN);
  var N: vec3f = select((*si).normal, (*si).faceNormal, useFaceN);

  (*ma).inside = ndv <= 0.0 && !(*ma).thinWalled;
  // In or out
  (*ma).relativeIor = select((*ma).specularIor, (1.0 / (*ma).specularIor), ndv > 0.0);
  var ffnormal = select(-N, N, ndv >= 0.0);
${hasCoatNormals ? /* wgsl */ `
  if ((*ma).coatNormalEnabled) {
    let view = -(*path).ray.d;
    let geometricView = dot(originalFace, view);
    (*ma).geometricNormal = select(-originalFace, originalFace, geometricView >= 0.0);
    ffnormal = coat_frame_normal(originalNormal, originalFace, view);
    (*ma).coatNormal = coat_frame_normal((*si).coatNormal, originalFace, view);
    (*ma).inside = geometricView <= 0.0 && !(*ma).thinWalled;
    (*ma).relativeIor = select((*ma).specularIor, 1.0 / (*ma).specularIor, geometricView > 0.0 || (*ma).thinWalled);
  }
` : ''}

  (*path).scatteringIsDelta = false;
  var mediumSampled = false;
  var crossedBoundary = false;
  let mediumType = (*si).mediumType;

  var scatteringPdf: f32;
  var scatterDir: vec3f;

  if (HAS_MEDIUM) {
    let d = distance((*si).position, (*path).ray.o);
    let segmentOrigin = (*path).ray.o;
    let extinction = (*si).mediumExtinction;
    let scattering = min((*si).mediumScattering, extinction);
    let majorant = max(max(extinction.x, extinction.y), extinction.z);
    if (mediumType > 0 && majorant > 0.0) {
      // Sample one RGB extinction channel and divide by the mixture PDF. This
      // avoids high-variance majorant weights for channel-dependent SSS radii.
      let mediumChannel = min(i32(sample1D(bounceDim(MEDIUM_CHANNEL)) * 3.0), 2);
      let channelExtinction = extinction[mediumChannel];
      let scatterDist = select(
        d,
        min(-log(max(sample1D(bounceDim(MEDIUM_DISTANCE)), 1e-7)) / channelExtinction, d),
        channelExtinction > 0.0
      );
      mediumSampled = scatterDist < d;
      let transmittance = exp(-extinction * scatterDist);
      let density = select(
        transmittance,
        extinction * transmittance,
        mediumSampled
      );
      let mediumPdf = max((density.x + density.y + density.z) / 3.0, 1e-8);
      if (mediumSampled) {
        (*path).beta *= transmittance * scattering / mediumPdf;
        // Move from the start of the segment, not from its end surface.
        (*path).ray.o = segmentOrigin + scatterDist * (*path).ray.d;
        (*si).position = (*path).ray.o;
      }
      else {
        (*path).beta *= transmittance / mediumPdf;
      }
    }
  }

  if (!mediumSampled) {
    (*path).li += emissiveHitMisWeight(si, path) * (*ma).emission * (*path).beta;
  }
  if ((*ma).materialType == UNLIT) {
    (*path).abort = true;
    return;
  }
  // Surface and medium events share the light estimator. Emission is added
  // before NEE, then the continuation direction is sampled.
  var prepared: OpenPbrPreparedBsdf;
  if (!mediumSampled) {
${hasAnisotropyDirections ? /* wgsl */ `
    (*ma).directionTangent = (*si).directionTangent;
    (*ma).directionBitangent = (*si).directionBitangent;
    (*ma).geometryTangent = (*si).geometryTangent;
    (*ma).geometryBitangent = (*si).geometryBitangent;
` : /* wgsl */ ``}
    prepared = openpbr_prepare((*si).tangent, *ma, -(*path).ray.d, ffnormal);
  }
  if (HAS_ENV_PORTALS) {
    portalGuideTotal = select(0.0, envPortalGuideTotal((*si).position, *ma), !mediumSampled);
  }
  DirectLight(si, *ma, prepared, path, ffnormal, !mediumSampled);
  if (mediumSampled) {
    // Pick a new direction based on the phase function.
    scatterDir = sampling_henyey_greenstein(
      -(*path).ray.d,
      (*si).mediumAnisotropy,
      sample2D(bounceDim(PHASE_DIRECTION))
    );
    scatteringPdf = sampling_henyey_greenstein_phase(
      dot(scatterDir, -(*path).ray.d),
      (*si).mediumAnisotropy
    );
  }
  else {
    // Sample BSDF for color and outgoing direction
    var isNextRefraction: bool;
    var sampledLobeRoughness = 1.0;
    var sampledLobeSpecular = false;
    var sampledLobeDiffuse = true;
    var sampledLobeIndex = u32(OPENPBR_LOBE_DIFFUSE);
    var sampledMarginalPdf = 0.0;
    let matRnd = vec3f(sample2D(bounceDim(BSDF_DIRECTION)), sample1D(bounceDim(BSDF_LOBE)));
    var fullResponse = vec3f(0.0);
    var f: vec3f;
    // A guided vertex aims a share of its continuations through the portals (salt "PGui")
    let guided = HAS_ENV_PORTALS && portalGuideTotal > 0.0;
    let portalChosen = guided && lightCandidateRandom(0x50477569u) < envPortals.regionCorner.w;
    if (portalChosen) {
      scatterDir = sampleEnvPortalDirection((*si).position, matRnd.xy, portalGuideTotal);
      f = openpbr_eval_prepared(prepared, scatterDir, &scatteringPdf);
      isNextRefraction = false;
      if (max(max(f.x, f.y), f.z) <= 0.0) {
        (*path).scatteringPdf = 0.0;
        (*path).abort = true;
        return;
      }
    } else {
      f = openpbr_sample_prepared(
        prepared,
        matRnd,
        &scatterDir,
        &scatteringPdf,
        &isNextRefraction,
        &sampledLobeRoughness,
        &sampledLobeSpecular,
        &sampledLobeDiffuse,
        &sampledLobeIndex,
        &sampledMarginalPdf,
        &fullResponse
      );
    }
    if (guided && (portalChosen || scatteringPdf > 0.0)) {
      scatteringPdf = envPortalGuidedPdf((*si).position, scatterDir, scatteringPdf);
    }
    // A rejected microfacet proposal (including failed refraction), or a
    // material with no scattering lobes, has no continuation. Keep emission
    // and NEE already accumulated at this vertex, but never trace a default
    // direction with the incoming throughput: that invents extra light paths.
    if (!(scatteringPdf > 0.0) ||
        !all(f == f) || !all(abs(f) < vec3f(3.402823e38)) ||
        dot(scatterDir, scatterDir) < 0.5) {
      (*path).scatteringPdf = 0.0;
      (*path).abort = true;
      return;
    }
    // Exactly index-matched solid transmission is a discrete null boundary;
    // environment NEE has no competing density for that event.
    (*path).scatteringIsDelta = !prepared.material.thinWalled &&
      abs(prepared.material.relativeIor - 1.0) <= 1e-6 &&
      prepared.probabilities[OPENPBR_LOBE_TRANSMISSION] >= 1.0 - 1e-6;
    (*path).beta *= f / scatteringPdf;

    crossedBoundary = isNextRefraction && !(*ma).thinWalled;
    (*path).isNextRefraction = isNextRefraction;
    // TODO prefer the opaque surface when transmission surface and opaque surface are overlapped.
    // Or the ray go out from transmission material will miss the opque surface.
    (*path).ray.o = (*si).position + EPS * scatterDir;
${hasCoatNormals ? /* wgsl */ `
    if ((*ma).coatNormalEnabled) { (*path).ray.o = (*si).position + EPS * select(-originalFace, originalFace, dot(originalFace, scatterDir) >= 0.0); }
` : ''}
  }
  (*path).scatteringPdf = scatteringPdf;
  (*path).ray.d = scatterDir;
  (*path).paneCrossed = false;
  if (mediumSampled) {
    (*path).isNextRefraction = false;
  }

  (*path).fromEye = false;

  if (HAS_MEDIUM) {
    // Reflections stay in the current medium. Enter or leave only after the
    // BSDF actually sampled transmission through a medium boundary.
    if (!mediumSampled && crossedBoundary && (*ma).mediumType > 0) {
      if (ndv1 >= 0.0) {
        (*si).mediumType = (*ma).mediumType;
        (*si).mediumExtinction = (*ma).mediumExtinction;
        (*si).mediumScattering = (*ma).mediumScattering;
        (*si).mediumAnisotropy = (*ma).mediumAnisotropy;
      }
      else {
        (*si).mediumType = 0;
        (*si).mediumExtinction = vec3f(0.0);
        (*si).mediumScattering = vec3f(0.0);
        (*si).mediumAnisotropy = 0.0;
      }
    }
  }
}


// A room photo's room: each directional light (the sun) takes its own shadow ray, a delta light needing no MIS, so
// the light RIS spends no candidate on a sun the room's walls hide; the room's table draws the rest. Where the sun's
// direction misses every portal the walls almost always hide it: its ray is traced with probability room.w only,
// and weighed up by as much (roulette, unbiased; sunlight through another room's window and a door still counts).
fn DirectSunInRoom(
  si: ptr<function, SurfaceInteraction>,
  ma: OpenPbrResolvedMaterial,
  prepared: OpenPbrPreparedBsdf,
  path: ptr<function, Path>,
  ffnormal: vec3f,
  isSurface: bool
) {
  for (var i = 0; i < rtUniform.lightsCount; i++) {
    let light = analyticLights[i];
    if (i32(light.litType) != DIRECTIONAL_LIGHT) {
      continue;
    }
    var sunSample: LightSampleRec;
    light_sample_directional(light, &sunSample);
    var roulette = 1.0;
    if (!envPortalsPassed((*si).position, sunSample.direction)) {
      if (lightCandidateRandom(0x53756e00u + u32(i)) >= envPortals.room.w) {
        continue;
      }
      roulette = 1.0 / envPortals.room.w;
    }
    sunSample.sourceType = 2u;
    sunSample.sourceIndex = u32(i);
    sunSample.selectionPmf = 1.0;
    var sunWork: ShadowRayWork;
    if (PrepareLightSample(si, ma, prepared, path, ffnormal, isSurface, sunSample, &sunWork) &&
        !intersectSceneShadow(sunWork.ray, sunWork.maxDistance, sunWork.triangleIndex)) {
      (*path).li += sunWork.contribution * roulette * shadowPaneTransmittance;
    }
  }
}
`;

}

export default createSampleMaterialWgsl();
