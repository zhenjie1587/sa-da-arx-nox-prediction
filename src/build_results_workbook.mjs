import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = ".";
const tablesDir = `${projectRoot}/results/tables`;
const outputPath = `${projectRoot}/results/results_summary.xlsx`;
const previewDir = `${projectRoot}/quality_checks/workbook_preview`;

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");

const imports = [
  ["Clean Comparison", "clean_model_comparison.csv"],
  ["Robustness", "robustness_summary.csv"],
  ["Regime", "regime_summary.csv"],
  ["Ablation", "ablation_study.csv"],
  ["Conformal", "conformal_summary.csv"],
  ["Lags", "selected_lags.csv"],
  ["Importance", "feature_importance.csv"],
  ["Statistics", "statistical_tests.csv"],
  ["Runtime", "model_runtime.csv"],
  ["Data Quality", "data_quality.csv"],
  ["Data Dictionary", "data_dictionary.csv"],
];

for (const [sheetName, fileName] of imports) {
  const csvText = await fs.readFile(`${tablesDir}/${fileName}`, "utf8");
  const importedWorkbook = await Workbook.fromCSV(csvText, { sheetName });
  const importedSheet = importedWorkbook.worksheets.getItem(sheetName);
  const importedRange = importedSheet.getUsedRange(true);
  const targetSheet = workbook.worksheets.add(sheetName);
  if (importedRange) {
    targetSheet
      .getRangeByIndexes(
        0,
        0,
        importedRange.rowCount,
        importedRange.columnCount,
      )
      .writeValues(importedRange.values);
  }
}

function rowsFromSheet(sheetName) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const values = sheet.getUsedRange(true).values;
  const headers = values[0].map((value) => String(value));
  return values.slice(1).map((row) =>
    Object.fromEntries(headers.map((header, index) => [header, row[index]])),
  );
}

const cleanRows = rowsFromSheet("Clean Comparison");
const robustnessRows = rowsFromSheet("Robustness");
const conformalRows = rowsFromSheet("Conformal");

const proposedClean = cleanRows.find(
  (row) => row.model === "SA-DA-ARX (proposed)",
);
const sourceLstm = cleanRows.find((row) => row.model === "Source-LSTM");
const proposedMcar30 = robustnessRows.find(
  (row) =>
    row.model === "SA-DA-ARX (proposed)" && row.scenario === "mcar30",
);
const baselineMcar30 = robustnessRows.find(
  (row) => row.model === "DA-ARX-Ridge" && row.scenario === "mcar30",
);
const cleanGain =
  100 * (1 - Number(proposedClean.rmse_mean) / Number(sourceLstm.rmse_mean));
const missingGain =
  100 *
  (1 - Number(proposedMcar30.rmse_mean) / Number(baselineMcar30.rmse_mean));

overview.showGridLines = false;
overview.getRange("A1:F2").merge();
overview.getRange("A1").values = [
  ["660-MW Coal-Fired Boiler NOx Robust Dynamic Prediction"],
];
overview.getRange("A1:F2").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 18 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
overview.getRange("A3:F3").merge();
overview.getRange("A3").values = [
  [
    "Chronological validation • sensor-missingness stress tests • blockwise conformal uncertainty",
  ],
];
overview.getRange("A3:F3").format = {
  fill: "#D9EAF7",
  font: { color: "#17365D", italic: true },
  horizontalAlignment: "center",
};

overview.getRange("A5:B5").merge();
overview.getRange("C5:D5").merge();
overview.getRange("E5:F5").merge();
overview.getRange("A5").values = [["Clean RMSE"]];
overview.getRange("C5").values = [["30% missing RMSE"]];
overview.getRange("E5").values = [["Clean R²"]];
overview.getRange("A6:B7").merge();
overview.getRange("C6:D7").merge();
overview.getRange("E6:F7").merge();
overview.getRange("A6").values = [[Number(proposedClean.rmse_mean)]];
overview.getRange("C6").values = [[Number(proposedMcar30.rmse_mean)]];
overview.getRange("E6").values = [[Number(proposedClean.r2_mean)]];
for (const cardRange of ["A5:B7", "C5:D7", "E5:F7"]) {
  overview.getRange(cardRange).format = {
    fill: "#EEF4FA",
    borders: { preset: "outside", style: "thin", color: "#9FBAD0" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
}
overview.getRange("A5:F5").format.font = {
  bold: true,
  color: "#3A5872",
};
overview.getRange("A6:F7").format.font = {
  bold: true,
  color: "#17365D",
  size: 16,
};
overview.getRange("A6:D7").format.numberFormat = '0.000 "mg/Nm³"';
overview.getRange("E6:F7").format.numberFormat = "0.0000";

overview.getRange("A10:B10").values = [["Model", "Clean RMSE"]];
const cleanTableRows = cleanRows.map((row) => [
  row.model,
  Number(row.rmse_mean),
]);
overview
  .getRangeByIndexes(10, 0, cleanTableRows.length, 2)
  .writeValues(cleanTableRows);
overview.getRange(`B11:B${10 + cleanTableRows.length}`).format.numberFormat =
  "0.000";

const scenarioOrder = ["clean", "mcar10", "mcar20", "mcar30"];
const scenarioLabels = ["Clean", "MCAR 10%", "MCAR 20%", "MCAR 30%"];
const robustnessModels = [
  "SA-DA-ARX (proposed)",
  "DA-ARX-Ridge",
  "Robust-DA-ARX",
];
overview
  .getRangeByIndexes(9, 3, 1, 4)
  .writeValues([["Scenario", ...robustnessModels]]);
const robustnessTableRows = scenarioOrder.map((scenario, scenarioIndex) => [
  scenarioLabels[scenarioIndex],
  ...robustnessModels.map((model) => {
    const row = robustnessRows.find(
      (candidate) =>
        candidate.model === model && candidate.scenario === scenario,
    );
    return Number(row.rmse_mean);
  }),
]);
overview
  .getRangeByIndexes(10, 3, robustnessTableRows.length, 4)
  .writeValues(robustnessTableRows);
overview.getRange("E11:G14").format.numberFormat = "0.000";

overview.getRange("A26:D26").values = [
  ["Scenario", "Coverage", "PINAW", "Half-width"],
];
const intervalOrder = [
  "clean",
  "mcar10",
  "mcar20",
  "mcar30",
  "block20",
  "critical_outage",
];
const intervalRows = intervalOrder.map((scenario) => {
  const row = conformalRows.find(
    (candidate) =>
      candidate.scenario === scenario && candidate.target === "Macro",
  );
  return [
    row.scenario_label,
    Number(row.picp_mean),
    Number(row.pinaw_mean),
    Number(row.quantile_half_width_mean),
  ];
});
overview.getRangeByIndexes(26, 0, intervalRows.length, 4).writeValues(intervalRows);
overview.getRange("B27:C32").format.numberFormat = "0.0%";
overview.getRange("D27:D32").format.numberFormat = "0.000";

overview.getRange("A35:F35").merge();
overview.getRange("A35").values = [["Key findings and provenance"]];
overview.getRange("A36:F40").values = [
  [
    "Result",
    `Proposed clean RMSE is ${cleanGain.toFixed(1)}% lower than the source-workbook LSTM.`,
    null,
    null,
    null,
    null,
  ],
  [
    "Robustness",
    `At 30% random missingness, RMSE is ${missingGain.toFixed(1)}% lower than the non-augmented DA-ARX model.`,
    null,
    null,
    null,
    null,
  ],
  [
    "Dataset",
    "https://data.mendeley.com/datasets/687fw2n74b/1",
    null,
    null,
    null,
    null,
  ],
  [
    "Source paper",
    "https://doi.org/10.1016/j.energy.2019.04.020",
    null,
    null,
    null,
    null,
  ],
  [
    "Important limitation",
    "F02–F27 are anonymized in the public workbook; exact plant-variable mapping must not be invented.",
    null,
    null,
    null,
    null,
  ],
];
for (let row = 36; row <= 40; row += 1) {
  overview.getRange(`B${row}:F${row}`).merge();
}
overview.getRange("A35:F35").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
};
overview.getRange("A36:A40").format.font = { bold: true, color: "#17365D" };
overview.getRange("A36:F40").format.wrapText = true;
overview.getRange("A36:F40").format.borders = {
  preset: "inside",
  style: "thin",
  color: "#D6DEE5",
};

const cleanChart = overview.charts.add("bar", overview.getRange(`A10:B${10 + cleanTableRows.length}`));
cleanChart.title = "Clean-sensor model comparison";
cleanChart.hasLegend = false;
cleanChart.yAxis = { numberFormatCode: "0.0" };
cleanChart.setPosition("H2", "P20");

const robustnessChart = overview.charts.add("line", overview.getRange("D10:G14"));
robustnessChart.title = "Random missingness robustness";
robustnessChart.hasLegend = true;
robustnessChart.yAxis = { numberFormatCode: "0.0" };
robustnessChart.setPosition("H22", "P41");

for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange(true);
  if (!used) continue;
  sheet.showGridLines = false;
  if (sheet.name !== "Overview") {
    sheet.freezePanes.freezeRows(1);
    const header = sheet.getRangeByIndexes(0, 0, 1, used.columnCount);
    header.format = {
      fill: "#17365D",
      font: { bold: true, color: "#FFFFFF" },
      wrapText: true,
      verticalAlignment: "center",
    };
    header.format.rowHeight = 32;
    const headers = header.values[0].map((value) => String(value));
    for (let col = 0; col < used.columnCount; col += 1) {
      const columnRange = sheet.getRangeByIndexes(
        0,
        col,
        used.rowCount,
        1,
      );
      const headerText = headers[col].toLowerCase();
      const isText =
        headerText.includes("model") ||
        headerText.includes("scenario") ||
        headerText.includes("target") ||
        headerText.includes("feature") ||
        headerText.includes("description") ||
        headerText.includes("status") ||
        headerText.includes("role") ||
        headerText.includes("unit");
      columnRange.format.columnWidth = isText ? 22 : 14;
      if (
        headerText.includes("rmse") ||
        headerText.includes("mae") ||
        headerText.includes("quantile") ||
        headerText.includes("seconds")
      ) {
        sheet
          .getRangeByIndexes(1, col, Math.max(1, used.rowCount - 1), 1)
          .format.numberFormat = "0.000";
      }
      if (
        headerText.includes("r2") ||
        headerText.includes("picp") ||
        headerText.includes("pinaw") ||
        headerText.includes("rate") ||
        headerText.includes("p_value")
      ) {
        sheet
          .getRangeByIndexes(1, col, Math.max(1, used.rowCount - 1), 1)
          .format.numberFormat = "0.0000";
      }
    }
    used.format.borders = {
      insideHorizontal: { style: "thin", color: "#E2E8F0" },
      bottom: { style: "thin", color: "#AAB7C4" },
    };
  }
}

overview.getRange("A1:P41").format.font.name = "Aptos";
overview.getRange("A1:A41").format.columnWidth = 20;
overview.getRange("B1:F41").format.columnWidth = 17;
overview.getRange("D10:D14").format.columnWidth = 18;
overview.freezePanes.freezeRows(3);

const inspectOverview = await workbook.inspect({
  kind: "table",
  range: "Overview!A1:P41",
  include: "values,formulas",
  tableMaxRows: 45,
  tableMaxCols: 16,
});
console.log(inspectOverview.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange(true);
  if (!used) continue;
  const previewRange = sheet.getRangeByIndexes(
    0,
    0,
    Math.min(used.rowCount, sheet.name === "Overview" ? 41 : 25),
    Math.min(used.columnCount, sheet.name === "Overview" ? 16 : 15),
  );
  const preview = await workbook.render({
    sheetName: sheet.name,
    range: previewRange.address,
    scale: 1,
    format: "png",
  });
  const safeName = sheet.name.replace(/[^A-Za-z0-9_-]+/g, "_");
  await fs.writeFile(
    `${previewDir}/${safeName}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Saved ${outputPath}`);
