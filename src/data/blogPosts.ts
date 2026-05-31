import { loadBlogPosts } from '@/lib/loadBlogs';

export interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  date: string;
  readTime: string;
  category: string;
  content: string;
}

// Load all blog posts from markdown files
export const blogPosts: BlogPost[] = loadBlogPosts();

export const getFeaturedPosts = () => blogPosts.slice(0, 2);
export const getPostById = (id: string) => blogPosts.find(post => post.id === id);
