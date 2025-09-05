// exporter.js (restored)
// Handles writing export JSON files.

const fs = require('fs');
const path = require('path');

function sanitizeFilename(name) {
    if (!name || typeof name !== 'string') return 'unnamed-group';
    let sanitized = name.replace(/[\u200E\u200F\u202A-\u202E\u2066-\u2069]/g, '');
    sanitized = sanitized.replace(/[\/\\?%*:|"<>]/g, '-');
    sanitized = sanitized.replace(/-+/g, '-').replace(/^-+|-+$/g, '');
    return sanitized || 'unnamed-group';
}

function writeExportFile(data, groupName, exportDir = 'exports') {
    if (!fs.existsSync(exportDir)) fs.mkdirSync(exportDir, { recursive: true });
    const filename = sanitizeFilename(groupName) + '.json';
    const outputPath = path.join(exportDir, filename);
    fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
    console.log(`✅ Exported ${data.messages.length} messages to ${outputPath}`);
    return outputPath;
}

module.exports = { writeExportFile, sanitizeFilename };
