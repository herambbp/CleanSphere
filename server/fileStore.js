import fs from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const NOTES_DIR = path.join(__dirname, '../notes');

// Get all courses
export async function getAllCourses() {
  try {
    const courses = await fs.readdir(NOTES_DIR);
    const courseData = [];

    for (const courseName of courses) {
      const coursePath = path.join(NOTES_DIR, courseName);
      const stat = await fs.stat(coursePath);

      if (stat.isDirectory()) {
        const chapters = await getChapters(courseName);
        courseData.push({
          id: courseName,
          name: formatName(courseName),
          chapters,
          totalNotes: chapters.reduce((acc, ch) => acc + ch.notes.length, 0),
        });
      }
    }
    return courseData;
  } catch (error) {
    console.error('Error reading courses:', error);
    return [];
  }
}

// Get course data with chapters
export async function getCourseData(courseId) {
  const chapters = await getChapters(courseId);
  return {
    id: courseId,
    name: formatName(courseId),
    chapters,
  };
}

// Get all chapters for a course
async function getChapters(courseId) {
  const coursePath = path.join(NOTES_DIR, courseId);
  const chapterDirs = await fs.readdir(coursePath);
  const chapters = [];

  for (const chapterName of chapterDirs) {
    const chapterPath = path.join(coursePath, chapterName);
    const stat = await fs.stat(chapterPath);

    if (stat.isDirectory()) {
      const notes = await getNotes(courseId, chapterName);
      chapters.push({
        id: chapterName,
        name: formatName(chapterName),
        notes,
      });
    }
  }
  return chapters;
}

// Get all notes in a chapter
async function getNotes(courseId, chapterId) {
  const chapterPath = path.join(NOTES_DIR, courseId, chapterId);
  const files = await fs.readdir(chapterPath);
  const notes = [];

  for (const file of files) {
    if (file.endsWith('.md')) {
      const filePath = path.join(chapterPath, file);
      const content = await fs.readFile(filePath, 'utf-8');
      const { data } = matter(content);

      notes.push({
        id: file.replace('.md', ''),
        title: data.title || formatName(file.replace('.md', '')),
        status: data.status || 'not-started',
        order: data.order || 0,
        readTime: data.readTime || calculateReadTime(content),
      });
    }
  }
  return notes.sort((a, b) => a.order - b.order);
}

// Get note content
export async function getNoteContent(courseId, chapterId, noteId) {
  const notePath = path.join(NOTES_DIR, courseId, chapterId, `${noteId}.md`);
  const content = await fs.readFile(notePath, 'utf-8');
  const { data, content: markdown } = matter(content);

  return {
    ...data,
    id: noteId,
    content: markdown,
    readTime: data.readTime || calculateReadTime(markdown),
  };
}

// Update note status
export async function updateNoteStatus(courseId, chapterId, noteId, status) {
  const notePath = path.join(NOTES_DIR, courseId, chapterId, `${noteId}.md`);
  const content = await fs.readFile(notePath, 'utf-8');
  const { data, content: markdown } = matter(content);

  data.status = status;
  const updated = matter.stringify(markdown, data);
  await fs.writeFile(notePath, updated);
}

// Save note content
export async function saveNoteContent(courseId, chapterId, noteId, newContent) {
  const notePath = path.join(NOTES_DIR, courseId, chapterId, `${noteId}.md`);
  await fs.writeFile(notePath, newContent);
}

// Helper: Format names
function formatName(str) {
  return str
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// Helper: Calculate read time
function calculateReadTime(content) {
  const wordsPerMinute = 200;
  const words = content.trim().split(/\s+/).length;
  const minutes = Math.ceil(words / wordsPerMinute);
  return `${minutes} min read`;
}
