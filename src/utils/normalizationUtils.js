// Normalization utilities for numerical data

// Min-Max Normalization: Scales values to [0, 1] range
export function minMaxNormalize(columnData) {
  const numericData = columnData.map(val => parseFloat(val)).filter(val => !isNaN(val));
  
  if (numericData.length === 0) {
    throw new Error('No valid numeric values found in column');
  }
  
  const min = Math.min(...numericData);
  const max = Math.max(...numericData);
  
  if (min === max) {
    // All values are the same, return array of 0s
    return {
      normalizedData: new Array(columnData.length).fill(0),
      min,
      max,
      type: 'minmax'
    };
  }
  
  const normalizedData = columnData.map(val => {
    const numVal = parseFloat(val);
    if (isNaN(numVal)) return val; // Keep non-numeric values as-is
    return (numVal - min) / (max - min);
  });
  
  return {
    normalizedData,
    min,
    max,
    type: 'minmax'
  };
}

// Z-Score Normalization: Standardizes values to mean=0, std=1
export function zScoreNormalize(columnData) {
  const numericData = columnData.map(val => parseFloat(val)).filter(val => !isNaN(val));
  
  if (numericData.length === 0) {
    throw new Error('No valid numeric values found in column');
  }
  
  // Calculate mean
  const mean = numericData.reduce((sum, val) => sum + val, 0) / numericData.length;
  
  // Calculate standard deviation
  const variance = numericData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / numericData.length;
  const stdDev = Math.sqrt(variance);
  
  if (stdDev === 0) {
    // All values are the same, return array of 0s
    return {
      normalizedData: new Array(columnData.length).fill(0),
      mean,
      stdDev: 0,
      type: 'zscore'
    };
  }
  
  const normalizedData = columnData.map(val => {
    const numVal = parseFloat(val);
    if (isNaN(numVal)) return val; // Keep non-numeric values as-is
    return (numVal - mean) / stdDev;
  });
  
  return {
    normalizedData,
    mean,
    stdDev,
    type: 'zscore'
  };
}

// Apply normalization to multiple columns of a dataset
export function normalizeDataset(rows, headers, normalizationConfig) {
  // normalizationConfig: { columnName: { type: 'minmax'|'zscore', params: {...} } }
  
  const normalizedRows = rows.map(row => [...row]); // Deep copy
  const normalizationInfo = {};
  
  // Apply normalization to each selected column
  Object.entries(normalizationConfig).forEach(([columnName, config]) => {
    const columnIndex = headers.indexOf(columnName);
    if (columnIndex === -1) return;
    
    // Extract column data
    const columnData = rows.map(row => row[columnIndex]);
    
    // Apply normalization
    let normalization;
    if (config.type === 'minmax') {
      normalization = minMaxNormalize(columnData);
    } else if (config.type === 'zscore') {
      normalization = zScoreNormalize(columnData);
    }
    
    // Replace column data with normalized values
    normalizedRows.forEach((row, rowIndex) => {
      row[columnIndex] = normalization.normalizedData[rowIndex];
    });
    
    // Store normalization info for this column
    normalizationInfo[columnName] = {
      type: config.type,
      min: normalization.min,
      max: normalization.max,
      mean: normalization.mean,
      stdDev: normalization.stdDev
    };
  });
  
  return {
    normalizedRows,
    normalizationInfo,
    headers: [...headers] // Headers remain the same
  };
}




