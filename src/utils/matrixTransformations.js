// Matrix transformation utilities for geometric operations

/**
 * Create 2D translation matrix
 */
export function translation2D(tx, ty) {
  return [
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
  ];
}

/**
 * Create 2D rotation matrix (angle in radians)
 */
export function rotation2D(angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    [cos, -sin, 0],
    [sin, cos, 0],
    [0, 0, 1]
  ];
}

/**
 * Create 2D scaling matrix
 */
export function scaling2D(sx, sy) {
  return [
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1]
  ];
}

/**
 * Create 2D reflection matrix (reflect across x-axis or y-axis)
 */
export function reflection2D(axis) {
  if (axis === 'x') {
    return [
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, 1]
    ];
  } else if (axis === 'y') {
    return [
      [-1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ];
  } else {
    // Reflect across origin
    return [
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, 1]
    ];
  }
}

/**
 * Create 2D shear matrix
 */
export function shear2D(shx, shy) {
  return [
    [1, shx, 0],
    [shy, 1, 0],
    [0, 0, 1]
  ];
}

/**
 * Transform a 2D point using homogeneous coordinates
 */
export function transformPoint2D(point, matrix) {
  const [x, y] = point;
  const homogeneous = [x, y, 1];
  const result = [0, 0, 0];
  
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      result[i] += matrix[i][j] * homogeneous[j];
    }
  }
  
  return [result[0], result[1]];
}

/**
 * Transform multiple points
 */
export function transformPoints2D(points, matrix) {
  return points.map(point => transformPoint2D(point, matrix));
}

/**
 * Create 3D translation matrix
 */
export function translation3D(tx, ty, tz) {
  return [
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]
  ];
}

/**
 * Create 3D rotation matrix around X axis
 */
export function rotation3DX(angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    [1, 0, 0, 0],
    [0, cos, -sin, 0],
    [0, sin, cos, 0],
    [0, 0, 0, 1]
  ];
}

/**
 * Create 3D rotation matrix around Y axis
 */
export function rotation3DY(angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    [cos, 0, sin, 0],
    [0, 1, 0, 0],
    [-sin, 0, cos, 0],
    [0, 0, 0, 1]
  ];
}

/**
 * Create 3D rotation matrix around Z axis
 */
export function rotation3DZ(angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    [cos, -sin, 0, 0],
    [sin, cos, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
}

/**
 * Create 3D scaling matrix
 */
export function scaling3D(sx, sy, sz) {
  return [
    [sx, 0, 0, 0],
    [0, sy, 0, 0],
    [0, 0, sz, 0],
    [0, 0, 0, 1]
  ];
}

/**
 * Transform a 3D point using homogeneous coordinates
 */
export function transformPoint3D(point, matrix) {
  const [x, y, z] = point;
  const homogeneous = [x, y, z, 1];
  const result = [0, 0, 0, 0];
  
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      result[i] += matrix[i][j] * homogeneous[j];
    }
  }
  
  return [result[0], result[1], result[2]];
}

/**
 * Project 3D point to 2D (simple perspective projection)
 */
export function project3DTo2D(point, distance = 5) {
  const [x, y, z] = point;
  const d = distance;
  return [
    (x * d) / (z + d),
    (y * d) / (z + d)
  ];
}

/**
 * Multiply two transformation matrices
 */
export function multiplyTransformMatrices(matrixA, matrixB) {
  const n = matrixA.length;
  const result = [];
  
  for (let i = 0; i < n; i++) {
    result[i] = [];
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += matrixA[i][k] * matrixB[k][j];
      }
      result[i][j] = sum;
    }
  }
  
  return result;
}

/**
 * Generate sample 2D shape (square)
 */
export function generateSquare2D(size = 1, centerX = 0, centerY = 0) {
  const half = size / 2;
  return [
    [centerX - half, centerY - half],
    [centerX + half, centerY - half],
    [centerX + half, centerY + half],
    [centerX - half, centerY + half],
    [centerX - half, centerY - half] // Close the shape
  ];
}

/**
 * Generate sample 3D shape (cube)
 */
export function generateCube3D(size = 1, centerX = 0, centerY = 0, centerZ = 0) {
  const half = size / 2;
  return [
    // Front face
    [centerX - half, centerY - half, centerZ + half],
    [centerX + half, centerY - half, centerZ + half],
    [centerX + half, centerY + half, centerZ + half],
    [centerX - half, centerY + half, centerZ + half],
    [centerX - half, centerY - half, centerZ + half],
    // Back face
    [centerX - half, centerY - half, centerZ - half],
    [centerX + half, centerY - half, centerZ - half],
    [centerX + half, centerY + half, centerZ - half],
    [centerX - half, centerY + half, centerZ - half],
    [centerX - half, centerY - half, centerZ - half],
    // Connecting edges
    [centerX - half, centerY - half, centerZ + half],
    [centerX - half, centerY - half, centerZ - half],
    [centerX + half, centerY - half, centerZ + half],
    [centerX + half, centerY - half, centerZ - half],
    [centerX + half, centerY + half, centerZ + half],
    [centerX + half, centerY + half, centerZ - half],
    [centerX - half, centerY + half, centerZ + half],
    [centerX - half, centerY + half, centerZ - half]
  ];
}

