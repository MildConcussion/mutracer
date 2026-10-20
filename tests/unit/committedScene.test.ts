import { afterEach, expect, it, vi } from 'vitest';
import { CommittedScene } from '../../src/renderer/committedScene';
import type { Scene } from '../../src/scene/Scene';
import type { SceneChanges } from '../../src/renderer/sceneChangeDetector';

/** Like the browser's ImageData: data is a getter on the prototype, so assigning it throws. */
class FakeImageData {
  readonly #data: Uint8ClampedArray;
  constructor(data: Uint8ClampedArray, readonly width: number, readonly height: number) { this.#data = data; }
  get data() { return this.#data; }
}

const full: SceneChanges = { meshes: 'full', transforms: [], colors: [], geometries: [], materials: [], lights: false, background: true };
const update: SceneChanges = { ...full, meshes: 'none' };

afterEach(() => { vi.unstubAllGlobals(); });

it('snapshots ImageData texture sources as their own ImageData, refreshed in place', () => {
  vi.stubGlobal('ImageData', FakeImageData);
  const image = new FakeImageData(new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 128]), 2, 1);
  const scene = { meshes: [], lights: [], background: { image } } as unknown as Scene;
  const committed = new CommittedScene();

  const first = (committed.capture(scene, full).background as unknown as { image: FakeImageData }).image;
  expect(first).toBeInstanceOf(FakeImageData);
  expect(first).not.toBe(image);
  expect(first.data).not.toBe(image.data);
  expect(Array.from(first.data)).toEqual(Array.from(image.data));

  image.data[0] = 99;
  const second = (committed.capture(scene, update).background as unknown as { image: FakeImageData }).image;
  expect(second).toBe(first);
  expect(second.data[0]).toBe(99);
});
