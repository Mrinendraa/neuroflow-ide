// Minimal linear algebra utilities for small matrices

export function transpose(matrix) {
  return matrix[0].map((_, i) => matrix.map(row => row[i]));
}

export function multiply(A, B) {
  const rowsA = A.length, colsA = A[0].length;
  const rowsB = B.length, colsB = B[0].length;
  if (colsA !== rowsB) throw new Error('Incompatible matrix sizes');
  const C = Array.from({ length: rowsA }, () => Array(colsB).fill(0));
  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      const a = A[i][k];
      for (let j = 0; j < colsB; j++) {
        C[i][j] += a * B[k][j];
      }
    }
  }
  return C;
}

export function invert(matrix) {
  const n = matrix.length;
  const A = matrix.map(row => row.slice());
  const I = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (__, j) => (i === j ? 1 : 0)));

  for (let i = 0; i < n; i++) {
    // pivot
    let maxRow = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(A[r][i]) > Math.abs(A[maxRow][i])) maxRow = r;
    }
    if (Math.abs(A[maxRow][i]) < 1e-12) throw new Error('Matrix is singular');
    if (maxRow !== i) {
      [A[i], A[maxRow]] = [A[maxRow], A[i]];
      [I[i], I[maxRow]] = [I[maxRow], I[i]];
    }
    // normalize row
    const pivot = A[i][i];
    for (let j = 0; j < n; j++) {
      A[i][j] /= pivot;
      I[i][j] /= pivot;
    }
    // eliminate others
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = A[r][i];
      for (let j = 0; j < n; j++) {
        A[r][j] -= factor * A[i][j];
        I[r][j] -= factor * I[i][j];
      }
    }
  }
  return I;
}








