# 📚 Study Notebook & Coding Documentation Platform

A beautiful, cloud-ready personal study platform with a **StudyLink-style UI** for organizing and viewing Markdown notes with course hierarchy, progress tracking, and rich content rendering.

![Platform Preview](https://img.shields.io/badge/Status-Production%20Ready-green)
![License](https://img.shields.io/badge/License-MIT-blue)

## ✨ Features

### 🎨 **Beautiful UI (StudyLink-Inspired)**
- **Dark Sidebar Navigation** - Home, Courses, Profile with modern icons
- **Breadcrumb Navigation** - Clear hierarchical navigation
- **Course Tree Structure** - Collapsible chapters with nested subtopics
- **Progress Badges** - Completed / In Progress / Not Started indicators
- **Responsive Design** - Works on desktop, tablet, and mobile

### 📝 **Markdown-Powered Content**
- Notes stored as `.md` files with YAML frontmatter
- Full course hierarchy: **Course → Chapter → Subtopic**
- Rich content rendering with:
  - ✅ Syntax highlighting (Highlight.js)
  - ✅ Math equations (KaTeX)
  - ✅ Tables, lists, blockquotes (GFM)
  - ✅ Code blocks with language detection

### 🧭 **Smart Navigation**
- **Notebook Sidebar** - Full course content tree
- **Jump-To Sidebar** - Auto-generated from headings (h2, h3)
- **Smooth Scroll** - Seamless section navigation
- **Active Section Highlighting** - Know where you are

### 📊 **Progress Tracking**
- Mark notes as Completed / In Progress / Not Started
- Visual progress indicators
- Continue where you left off

### ✏️ **Built-in Editor**
- Edit notes directly in browser
- Save changes to filesystem
- Real-time preview

### ☁️ **Cloud-Ready**
- Deploy to **Vercel**, **Netlify**, or **Railway**
- Docker support for containerized deployment
- Easy to switch to S3/GitHub storage

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd study-notebook-platform

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will run at:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:5000

---

## 📁 Project Structure

```
study-notebook-platform/
├── notes/                      # Markdown notes organized by course
│   └── python/
│       ├── intro/
│       │   ├── getting-started.md
│       │   └── history-of-python.md
│       └── oop/
│           ├── classes-and-objects.md
│           └── inheritance.md
├── server/                     # Backend API
│   ├── index.js               # Express server
│   └── fileStore.js           # File-based storage
├── src/                       # Frontend React app
│   ├── components/
│   │   ├── Sidebar.jsx        # Left navigation
│   │   ├── CourseTree.jsx     # Expandable course chapters
│   │   ├── NotebookSidebar.jsx # Course content sidebar
│   │   ├── JumpToSidebar.jsx  # Jump-to headings
│   │   ├── MarkdownRenderer.jsx
│   │   └── ProgressBadge.jsx
│   ├── pages/
│   │   ├── HomePage.jsx       # Dashboard
│   │   ├── CoursePage.jsx     # Course overview
│   │   └── NotePage.jsx       # Reading view
│   ├── utils/
│   │   ├── fetcher.js         # API client
│   │   ├── calculateReadTime.js
│   │   └── mdParser.js
│   ├── App.jsx
│   └── main.jsx
├── public/
├── Dockerfile
├── docker-compose.yml
└── package.json
```

---

## 📝 Creating Notes

### Note Structure

Each note is a Markdown file with YAML frontmatter:

```markdown
---
title: History of Python
course: python
chapter: intro
order: 1
tags: [python, history]
status: not-started
readTime: 8 min read
---

# History of Python

Your content here...

## Section 1
Content...

## Section 2
Content...
```

### Frontmatter Fields

| Field | Description | Required |
|-------|-------------|----------|
| `title` | Note title | Yes |
| `course` | Course ID (folder name) | Yes |
| `chapter` | Chapter ID (folder name) | Yes |
| `order` | Display order (0, 1, 2...) | Yes |
| `tags` | Array of tags | No |
| `status` | `not-started`, `in-progress`, `completed` | No |
| `readTime` | Auto-calculated if omitted | No |

### Adding New Courses

1. Create folder structure:
```bash
mkdir -p notes/javascript/basics
```

2. Add note:
```bash
touch notes/javascript/basics/variables.md
```

3. Write content with frontmatter
4. Restart server to detect new course

---

## 🎨 Customization

### Tailwind Colors

Edit `tailwind.config.js`:

```javascript
colors: {
  primary: {
    500: '#0ea5e9',  // Change primary color
  },
  sidebar: {
    bg: '#1a1d29',    // Sidebar background
  },
}
```

### Typography

Customize in `src/components/MarkdownRenderer.jsx`

---

## 🐳 Docker Deployment

### Development

```bash
docker-compose --profile dev up
```

### Production

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## ☁️ Cloud Deployment

### Deploy to Vercel

1. **Install Vercel CLI**:
```bash
npm i -g vercel
```

2. **Deploy**:
```bash
vercel
```

3. **Configure**:
- Framework Preset: **Vite**
- Build Command: `npm run build`
- Output Directory: `dist`

### Deploy to Netlify

1. **Install Netlify CLI**:
```bash
npm i -g netlify-cli
```

2. **Deploy**:
```bash
netlify deploy --prod
```

3. **Build Settings**:
- Build Command: `npm run build`
- Publish Directory: `dist`

### Deploy to Railway

1. Create a new project on [Railway](https://railway.app)
2. Connect your GitHub repository
3. Set build command: `npm run build`
4. Railway will auto-detect Dockerfile

### Deploy to Render

1. Create a **Web Service** on [Render](https://render.com)
2. Connect your repository
3. Set:
   - Build Command: `npm run build`
   - Start Command: `npm start`

---

## 🔧 Advanced Configuration

### Using S3 for Storage

Replace `server/fileStore.js` with S3 integration:

```javascript
import AWS from 'aws-sdk';

const s3 = new AWS.S3({
  accessKeyId: process.env.AWS_ACCESS_KEY,
  secretAccessKey: process.env.AWS_SECRET_KEY,
});

export async function getNoteContent(courseId, chapterId, noteId) {
  const params = {
    Bucket: 'your-bucket',
    Key: `notes/${courseId}/${chapterId}/${noteId}.md`,
  };
  const data = await s3.getObject(params).promise();
  return data.Body.toString('utf-8');
}
```

### GitHub as Storage

Use GitHub API to read/write notes from a repository.

---

## 🛠️ API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/courses` | List all courses |
| GET | `/api/courses/:courseId` | Get course with chapters |
| GET | `/api/notes/:courseId/:chapterId/:noteId` | Get note content |
| POST | `/api/notes/status` | Update note status |
| POST | `/api/notes/save` | Save note content |

### Example Usage

```javascript
// Get all courses
const courses = await fetch('/api/courses').then(r => r.json());

// Get specific note
const note = await fetch('/api/notes/python/intro/history-of-python')
  .then(r => r.json());

// Update status
await fetch('/api/notes/status', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    courseId: 'python',
    chapterId: 'intro',
    noteId: 'history-of-python',
    status: 'completed'
  })
});
```

---

## 🎯 Development Scripts

```bash
# Run both frontend and backend
npm run dev

# Frontend only (dev mode)
npm run client

# Backend only
npm run server

# Build for production
npm run build

# Preview production build
npm run preview

# Start production server
npm start
```

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Kill process on port 3000
npx kill-port 3000

# Kill process on port 5000
npx kill-port 5000
```

### Notes Not Loading

1. Check file structure matches `notes/COURSE/CHAPTER/NOTE.md`
2. Verify frontmatter syntax
3. Restart server after adding new notes

### Styling Issues

```bash
# Rebuild Tailwind
npm run build
```

---

## 📄 License

MIT License - Feel free to use for personal or commercial projects.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📬 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues first

---

**Built with ❤️ using React, Express, TailwindCSS, and Markdown**

Enjoy your learning journey! 🚀
