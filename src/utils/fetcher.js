const API_BASE = '/api';

export const fetcher = {
  async getCourses() {
    const response = await fetch(`${API_BASE}/courses`);
    if (!response.ok) throw new Error('Failed to fetch courses');
    return response.json();
  },

  async getCourse(courseId) {
    const response = await fetch(`${API_BASE}/courses/${courseId}`);
    if (!response.ok) throw new Error('Failed to fetch course');
    return response.json();
  },

  async getNote(courseId, chapterId, noteId) {
    const response = await fetch(`${API_BASE}/notes/${courseId}/${chapterId}/${noteId}`);
    if (!response.ok) throw new Error('Failed to fetch note');
    return response.json();
  },

  async updateNoteStatus(courseId, chapterId, noteId, status) {
    const response = await fetch(`${API_BASE}/notes/status`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ courseId, chapterId, noteId, status }),
    });
    if (!response.ok) throw new Error('Failed to update status');
    return response.json();
  },

  async saveNote(courseId, chapterId, noteId, content) {
    const response = await fetch(`${API_BASE}/notes/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ courseId, chapterId, noteId, content }),
    });
    if (!response.ok) throw new Error('Failed to save note');
    return response.json();
  },
};
