import { Link } from "react-router-dom";
import { BlogPost } from "@/data/blogPosts";
import { Badge } from "@/components/ui/badge";

interface BlogCardProps {
  post: BlogPost;
}

const BlogCard = ({ post }: BlogCardProps) => {
  return (
    <Link to={`/blog/${post.id}`} className="group block">
      <article className="py-8 border-b border-border last:border-b-0 transition-all">
        <div className="flex items-center gap-3 mb-3">
          <Badge variant="secondary" className="text-xs">
            {post.category}
          </Badge>
          <span className="text-sm text-muted-foreground">{post.date}</span>
          <span className="text-sm text-muted-foreground">·</span>
          <span className="text-sm text-muted-foreground">{post.readTime}</span>
        </div>
        <h2 className="text-2xl font-serif font-bold mb-3 group-hover:text-accent transition-colors">
          {post.title}
        </h2>
        <p className="text-muted-foreground text-lg leading-relaxed">
          {post.excerpt}
        </p>
      </article>
    </Link>
  );
};

export default BlogCard;
