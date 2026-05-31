import { BlogPost } from '@/data/blogPosts';

// Simple browser-compatible frontmatter parser
function parseFrontmatter(markdown: string) {
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
  const match = markdown.match(frontmatterRegex);
  
  if (!match) {
    return { data: {}, content: markdown };
  }
  
  const [, frontmatterText, content] = match;
  const data: Record<string, string> = {};
  
  // Parse YAML-like frontmatter
  frontmatterText.split('\n').forEach(line => {
    const colonIndex = line.indexOf(':');
    if (colonIndex > 0) {
      const key = line.slice(0, colonIndex).trim();
      const value = line.slice(colonIndex + 1).trim().replace(/^["']|["']$/g, '');
      data[key] = value;
    }
  });
  
  return { data, content: content.trim() };
}

// Import all markdown files from the blogs directory
const blogFiles = import.meta.glob('/src/content/blogs/*.md', { 
  eager: true, 
  query: '?raw',
  import: 'default'
});

export function loadBlogPosts(): BlogPost[] {
  const posts: BlogPost[] = [];

  for (const path in blogFiles) {
    const markdown = blogFiles[path] as string;
    const { data, content } = parseFrontmatter(markdown);
    
    // Extract filename without extension to use as ID
    const filename = path.split('/').pop()?.replace('.md', '') || '';
    
    posts.push({
      id: filename,
      title: data.title || 'Untitled',
      excerpt: data.excerpt || '',
      date: data.date || new Date().toISOString().split('T')[0],
      readTime: data.readTime || '5 min read',
      category: data.category || 'Uncategorized',
      content: content
    });
  }

  // Sort by date (newest first)
  return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}
