import { useParams, Link } from "react-router-dom";
import Navigation from "@/components/Navigation";
import { getPostById } from "@/data/blogPosts";
import ReactMarkdown from "react-markdown";
import rehypeSlug from 'rehype-slug'; 
import remarkToc from 'remark-toc';
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft } from "lucide-react";

const BlogPost = () => {
  const { id } = useParams<{ id: string }>();
  const post = id ? getPostById(id) : null;

  if (!post) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <main className="container mx-auto px-4 py-12 max-w-4xl">
          <div className="text-center">
            <h1 className="text-4xl font-serif font-bold mb-4">Post not found</h1>
            <Link to="/blogs" className="text-accent hover:underline">
              Return to all posts
            </Link>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-4 py-12 max-w-3xl">
        <Link 
          to="/blogs" 
          className="inline-flex items-center gap-2 text-muted-foreground hover:text-accent transition-colors mb-8"
        >
          <ArrowLeft className="w-4 h-4" />
          All posts
        </Link>

        <article>
          <header className="mb-12">
            <div className="flex items-center gap-3 mb-4">
              <Badge variant="secondary">{post.category}</Badge>
              <span className="text-sm text-muted-foreground">{post.date}</span>
              <span className="text-sm text-muted-foreground">·</span>
              <span className="text-sm text-muted-foreground">{post.readTime}</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-serif font-bold mb-4">
              {post.title}
            </h1>
            <p className="text-xl text-muted-foreground">
              {post.excerpt}
            </p>
          </header>

          <div className="markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkMath, remarkGfm,remarkToc ]}
              rehypePlugins={[rehypeKatex,rehypeSlug]}
            >
              {post.content}
            </ReactMarkdown>
          </div>
        </article>

        <div className="mt-16 pt-8 border-t border-border">
          <Link 
            to="/blogs" 
            className="inline-flex items-center gap-2 text-accent hover:underline font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to all posts
          </Link>
        </div>
      </main>

      <footer className="border-t border-border mt-20">
        <div className="container mx-auto px-4 py-8 max-w-4xl">
          <p className="text-center text-muted-foreground">
            © 2024 ML & CS Insights. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default BlogPost;
