// common.js (restored)
// Utility helpers for ID and phone normalization and Hebrew text cleanup.

function hebrewifyIfNeeded(text) {
    if (!text || typeof text !== 'string') return text || '';
    // Strip Unicode direction/control marks; keep original order.
    return text.replace(/[\u200E\u200F\u202A-\u202E\u2066-\u2069]/g, '').trim();
}

function isLid(input) {
    return typeof input === 'string' && input.includes('@lid');
}

function isPhoneNumber(input) {
    if (typeof input !== 'string') return false;
    const phoneRegex = /(\+?972[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{4})|(972\d{9})/;
    return phoneRegex.test(input);
}

function normalizePhoneNumber(input) {
    if (typeof input !== 'string') return input;
    const normalized = input.replace(/[^\d]/g, '');
    return normalized.startsWith('972') ? normalized : input;
}

module.exports = {
    hebrewifyIfNeeded,
    isLid,
    isPhoneNumber,
    normalizePhoneNumber
};
