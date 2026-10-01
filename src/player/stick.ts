export interface StickReading {
  /** Movement: strafe right / forward in [-1, 1], after the dead zone. */
  right: number;
  forward: number;
  /** 0 inside the dead zone, 1 at full deflection. */
  magnitude: number;
  /** Knob offset from the stick's centre in CSS px (clamped to the radius). */
  knobX: number;
  knobY: number;
}

/**
 * Turn a drag from the point where a virtual stick was touched (dx, dy in CSS
 * px, screen coordinates, so +y is down) into an analog movement vector.
 * The knob stops at `radius`; drags shorter than `deadZone * radius` read as 0,
 * and the rest of the travel is rescaled so movement starts smoothly from 0.
 */
export function readStick(dx: number, dy: number, radius: number, deadZone: number): StickReading {
  const dist = Math.hypot(dx, dy);
  if (dist === 0) return { right: 0, forward: 0, magnitude: 0, knobX: 0, knobY: 0 };
  const ux = dx / dist, uy = dy / dist;
  const travel = Math.min(dist, radius);
  const raw = travel / radius;
  const magnitude = raw <= deadZone ? 0 : (raw - deadZone) / (1 - deadZone);
  return { right: ux * magnitude, forward: -uy * magnitude, magnitude, knobX: ux * travel, knobY: uy * travel };
}
