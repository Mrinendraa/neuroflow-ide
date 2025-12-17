// Encoding utilities for categorical data

// Label Encoding: Maps each unique value to an integer (0, 1, 2, ...)
export function labelEncode(columnData) {
  const uniqueValues = [...new Set(columnData)];
  const encodingMap = {};
  const reverseMap = {};
  
  uniqueValues.forEach((value, index) => {
    encodingMap[value] = index;
    reverseMap[index] = value;
  });
  
  const encodedData = columnData.map(value => encodingMap[value]);
  
  return {
    encodedData,
    encodingMap,
    reverseMap,
    type: 'label'
  };
}

// Frequency Encoding: Maps each value to its frequency in the dataset
export function frequencyEncode(columnData) {
  const frequencyMap = {};
  
  // Count frequencies
  columnData.forEach(value => {
    frequencyMap[value] = (frequencyMap[value] || 0) + 1;
  });
  
  const encodedData = columnData.map(value => frequencyMap[value]);
  
  return {
    encodedData,
    encodingMap: frequencyMap,
    reverseMap: null, // Not applicable for frequency encoding
    type: 'frequency'
  };
}

// Apply encoding to multiple columns of a dataset
export function encodeDataset(rows, headers, encodingConfig) {
  // encodingConfig: { columnName: { type: 'label'|'frequency', encoding: {...} } }
  
  const encodedRows = rows.map(row => [...row]); // Deep copy
  const encodingInfo = {};
  
  // Apply encoding to each selected column
  Object.entries(encodingConfig).forEach(([columnName, config]) => {
    const columnIndex = headers.indexOf(columnName);
    if (columnIndex === -1) return;
    
    // Extract column data
    const columnData = rows.map(row => row[columnIndex]);
    
    // Apply encoding
    let encoding;
    if (config.type === 'label') {
      encoding = labelEncode(columnData);
    } else if (config.type === 'frequency') {
      encoding = frequencyEncode(columnData);
    }
    
    // Replace column data with encoded values
    encodedRows.forEach((row, rowIndex) => {
      row[columnIndex] = encoding.encodedData[rowIndex];
    });
    
    // Store encoding info for this column
    encodingInfo[columnName] = {
      type: config.type,
      encodingMap: encoding.encodingMap,
      reverseMap: encoding.reverseMap
    };
  });
  
  return {
    encodedRows,
    encodingInfo,
    headers: [...headers] // Headers remain the same
  };
}




