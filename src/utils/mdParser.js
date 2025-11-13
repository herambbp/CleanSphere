/**
 * Parse markdown frontmatter and content
 * @param {string} markdown - Raw markdown string
 * @returns {Object} Parsed data and content
 */
export function parseFrontmatter(markdown) {
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
  const match = markdown.match(frontmatterRegex);

  if (!match) {
    return { data: {}, content: markdown };
  }

  const [, frontmatter, content] = match;
  const data = {};

  frontmatter.split('\n').forEach(line => {
    const [key, ...values] = line.split(':');
    if (key && values.length) {
      const value = values.join(':').trim();
      data[key.trim()] = value.replace(/^['"]|['"]$/g, '');
    }
  });

  return { data, content: content.trim() };
}

/**
 * Generate slug from text
 * @param {string} text - Text to slugify
 * @returns {string} Slug
 */
export function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .trim();
}
