import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import {
  getAllCourses,
  getCourseData,
  getNoteContent,
  updateNoteStatus,
  saveNoteContent
} from './fileStore.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.get('/api/courses', async (req, res) => {
  try {
    const courses = await getAllCourses();
    res.json(courses);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/courses/:courseId', async (req, res) => {
  try {
    const { courseId } = req.params;
    const courseData = await getCourseData(courseId);
    res.json(courseData);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/notes/:courseId/:chapterId/:noteId', async (req, res) => {
  try {
    const { courseId, chapterId, noteId } = req.params;
    const note = await getNoteContent(courseId, chapterId, noteId);
    res.json(note);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/notes/status', async (req, res) => {
  try {
    const { courseId, chapterId, noteId, status } = req.body;
    await updateNoteStatus(courseId, chapterId, noteId, status);
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/notes/save', async (req, res) => {
  try {
    const { courseId, chapterId, noteId, content } = req.body;
    await saveNoteContent(courseId, chapterId, noteId, content);
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Serve static files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(join(__dirname, '../dist')));
  app.get('*', (req, res) => {
    res.sendFile(join(__dirname, '../dist/index.html'));
  });
}

app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
