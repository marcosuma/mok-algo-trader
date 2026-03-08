// Polyfill for Node 16: patch node:crypto to expose webcrypto methods directly,
// matching the Node 18 API surface that vite's bundled code expects.
const cryptoMod = require('crypto')
if (typeof cryptoMod.getRandomValues !== 'function' && cryptoMod.webcrypto) {
  cryptoMod.getRandomValues = (arr) => cryptoMod.webcrypto.getRandomValues(arr)
}
if (!globalThis.crypto) {
  globalThis.crypto = cryptoMod.webcrypto
}
