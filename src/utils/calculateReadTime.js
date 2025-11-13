/**
 * Calculate estimated read time for content
 * @param {string} content - The markdown content
 * @returns {string} Read time string (e.g., "5 min read")
 */
export function calculateReadTime(content) {
  const wordsPerMinute = 200;
  const words = content.trim().split(/\s+/).length;
  const minutes = Math.ceil(words / wordsPerMinute);
  return `${minutes} min read`;
}

/**
 * Extract headings from markdown content
 * @param {string} content - The markdown content
 * @returns {Array} Array of heading objects with level, text, and id
 */
export function extractHeadings(content) {
  const headingRegex = /^(#{2,3})\s+(.+)$/gm;
  const headings = [];
  let match;

  while ((match = headingRegex.exec(content)) !== null) {
    const level = match[1].length;
    const text = match[2].trim();
    const id = text.toLowerCase().replace(/[^\w]+/g, '-');

    headings.push({ level, text, id });
  }

  return headings;
}
