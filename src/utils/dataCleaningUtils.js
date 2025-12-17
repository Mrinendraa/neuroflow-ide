/**
 * Data Cleaning Utilities
 * Comprehensive functions for cleaning and preprocessing data
 */

/**
 * Check if a value is missing/null
 */
export const isMissing = (value) => {
    return value === null || value === undefined || value === '' ||
        (typeof value === 'number' && isNaN(value));
};

/**
 * Check if a value is numeric
 */
export const isNumeric = (value) => {
    if (isMissing(value)) return false;
    return !isNaN(parseFloat(value)) && isFinite(value);
};

/**
 * Calculate statistics for a column
 */
export const calculateColumnStats = (values) => {
    const numericValues = values
        .filter(v => !isMissing(v) && isNumeric(v))
        .map(v => parseFloat(v));

    if (numericValues.length === 0) {
        return { mean: 0, median: 0, mode: 0, min: 0, max: 0, stdDev: 0, q1: 0, q3: 0 };
    }

    // Sort for median and quartiles
    const sorted = [...numericValues].sort((a, b) => a - b);

    // Mean
    const sum = sorted.reduce((a, b) => a + b, 0);
    const mean = sum / sorted.length;

    // Median
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];

    // Mode (most frequent value)
    const counts = {};
    let maxCount = 0;
    let mode = sorted[0];
    for (const v of sorted) {
        counts[v] = (counts[v] || 0) + 1;
        if (counts[v] > maxCount) {
            maxCount = counts[v];
            mode = v;
        }
    }

    // Min and Max
    const min = sorted[0];
    const max = sorted[sorted.length - 1];

    // Standard Deviation
    const variance = sorted.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / sorted.length;
    const stdDev = Math.sqrt(variance);

    // Quartiles
    const q1Index = Math.floor(sorted.length * 0.25);
    const q3Index = Math.floor(sorted.length * 0.75);
    const q1 = sorted[q1Index];
    const q3 = sorted[q3Index];

    return { mean, median, mode, min, max, stdDev, q1, q3 };
};

/**
 * Calculate mode for categorical data
 */
export const calculateCategoricalMode = (values) => {
    const nonMissing = values.filter(v => !isMissing(v));
    if (nonMissing.length === 0) return null;

    const counts = {};
    let maxCount = 0;
    let mode = nonMissing[0];

    for (const v of nonMissing) {
        const key = String(v);
        counts[key] = (counts[key] || 0) + 1;
        if (counts[key] > maxCount) {
            maxCount = counts[key];
            mode = v;
        }
    }

    return mode;
};

/**
 * Handle missing values in dataset
 */
export const handleMissingValues = (rows, headers, config) => {
    const { method } = config; // 'drop', 'fill_mean', 'fill_median', 'fill_mode'

    let processedRows = rows.map(row => [...row]);
    let stats = {};

    if (method === 'drop') {
        // Drop rows with any missing values
        processedRows = processedRows.filter(row =>
            row.every(cell => !isMissing(cell))
        );
    } else {
        // Calculate stats for each column
        headers.forEach((header, colIdx) => {
            const columnValues = rows.map(row => row[colIdx]);
            const isNumericColumn = columnValues.some(v => !isMissing(v) && isNumeric(v));

            if (isNumericColumn) {
                stats[colIdx] = calculateColumnStats(columnValues);
            } else {
                stats[colIdx] = { mode: calculateCategoricalMode(columnValues) };
            }
        });

        // Fill missing values
        processedRows = processedRows.map(row =>
            row.map((cell, colIdx) => {
                if (isMissing(cell)) {
                    const columnValues = rows.map(row => row[colIdx]);
                    const isNumericColumn = columnValues.some(v => !isMissing(v) && isNumeric(v));

                    if (method === 'fill_mean' && isNumericColumn) {
                        return stats[colIdx].mean;
                    } else if (method === 'fill_median' && isNumericColumn) {
                        return stats[colIdx].median;
                    } else if (method === 'fill_mode') {
                        return stats[colIdx].mode;
                    }
                    return 0; // Default fallback
                }
                return cell;
            })
        );
    }

    return {
        rows: processedRows,
        stats,
        removedCount: rows.length - processedRows.length
    };
};

/**
 * Remove outliers using different methods
 */
export const removeOutliers = (rows, headers, config) => {
    const { method, threshold, numericColumns } = config;
    // method: 'iqr', 'zscore', 'percentile'
    // threshold: 1.5 for IQR, 3 for Z-score, or percentile values

    if (!numericColumns || numericColumns.length === 0) {
        return { rows, removedCount: 0, outlierInfo: {} };
    }

    let processedRows = [...rows];
    const outlierInfo = {};

    numericColumns.forEach(colIdx => {
        const columnValues = processedRows
            .map(row => row[colIdx])
            .filter(v => !isMissing(v) && isNumeric(v))
            .map(v => parseFloat(v));

        if (columnValues.length < 4) return; // Not enough data

        const stats = calculateColumnStats(columnValues);
        let lowerBound, upperBound;

        if (method === 'iqr') {
            const iqr = stats.q3 - stats.q1;
            lowerBound = stats.q1 - threshold * iqr;
            upperBound = stats.q3 + threshold * iqr;
        } else if (method === 'zscore') {
            lowerBound = stats.mean - threshold * stats.stdDev;
            upperBound = stats.mean + threshold * stats.stdDev;
        } else if (method === 'percentile') {
            const sorted = [...columnValues].sort((a, b) => a - b);
            const lowerIndex = Math.floor(sorted.length * (threshold / 100));
            const upperIndex = Math.floor(sorted.length * ((100 - threshold) / 100));
            lowerBound = sorted[lowerIndex];
            upperBound = sorted[upperIndex];
        }

        outlierInfo[headers[colIdx]] = { lowerBound, upperBound, method };

        // Filter rows
        processedRows = processedRows.filter(row => {
            const value = parseFloat(row[colIdx]);
            if (isMissing(row[colIdx]) || !isNumeric(row[colIdx])) return true;
            return value >= lowerBound && value <= upperBound;
        });
    });

    return {
        rows: processedRows,
        removedCount: rows.length - processedRows.length,
        outlierInfo
    };
};

/**
 * Remove duplicate rows
 */
export const removeDuplicates = (rows, config) => {
    const { keepOption } = config; // 'first' or 'last'

    const seen = new Set();
    const processedRows = [];

    const rowsToProcess = keepOption === 'last' ? [...rows].reverse() : rows;

    for (const row of rowsToProcess) {
        const rowKey = JSON.stringify(row);
        if (!seen.has(rowKey)) {
            seen.add(rowKey);
            processedRows.push([...row]);
        }
    }

    const finalRows = keepOption === 'last' ? processedRows.reverse() : processedRows;

    return {
        rows: finalRows,
        removedCount: rows.length - finalRows.length
    };
};

/**
 * Main cleaning function that orchestrates all operations
 */
export const cleanDataset = (rows, headers, config) => {
    let processedRows = rows.map(row => [...row]);
    const cleaningLog = [];
    const cleaningStats = {};

    // 1. Handle Missing Values
    if (config.handleMissing) {
        const result = handleMissingValues(processedRows, headers, {
            method: config.missingMethod
        });
        processedRows = result.rows;
        cleaningStats.missingStats = result.stats;
        if (config.missingMethod === 'drop') {
            cleaningLog.push(`Dropped ${result.removedCount} rows with missing values`);
        } else {
            cleaningLog.push(`Filled missing values using ${config.missingMethod.replace('fill_', '')}`);
        }
    }

    // 2. Remove Outliers
    if (config.removeOutliers && config.outlierColumns?.length > 0) {
        const result = removeOutliers(processedRows, headers, {
            method: config.outlierMethod,
            threshold: config.outlierThreshold,
            numericColumns: config.outlierColumns
        });
        processedRows = result.rows;
        cleaningStats.outlierInfo = result.outlierInfo;
        cleaningLog.push(`Removed ${result.removedCount} outlier rows using ${config.outlierMethod} method`);
    }

    // 3. Remove Duplicates
    if (config.removeDuplicates) {
        const result = removeDuplicates(processedRows, {
            keepOption: config.duplicateKeepOption
        });
        processedRows = result.rows;
        cleaningLog.push(`Removed ${result.removedCount} duplicate rows (kept ${config.duplicateKeepOption})`);
    }

    return {
        cleanedRows: processedRows,
        cleaningLog,
        cleaningStats,
        originalRowCount: rows.length,
        cleanedRowCount: processedRows.length,
        removedRowCount: rows.length - processedRows.length
    };
};
