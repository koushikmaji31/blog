import Navigation from "@/components/Navigation";
import BlogCard from "@/components/BlogCard";
import { blogPosts } from "@/data/blogPosts";

const Blogs = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="container mx-auto px-4 py-12 max-w-4xl">
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl font-serif font-bold mb-4">All Posts</h1>
          <p className="text-lg text-muted-foreground">
            {blogPosts.length} articles on machine learning and computer science
          </p>
        </div>

        <div className="space-y-0">
          {blogPosts.map((post) => (
            <BlogCard key={post.id} post={post} />
          ))}
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

export default Blogs;
