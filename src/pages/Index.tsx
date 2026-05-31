import Navigation from "@/components/Navigation";
import BlogCard from "@/components/BlogCard";
import { getFeaturedPosts } from "@/data/blogPosts";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const Index = () => {
  const featuredPosts = getFeaturedPosts();

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 max-w-4xl">
        <div className="text-center space-y-6">
          <h1 className="text-5xl md:text-6xl font-serif font-bold tracking-tight">
            Machine Learning & Computer Science Insights
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Exploring the mathematical foundations and theoretical concepts behind modern computing and artificial intelligence
          </p>
          <div className="pt-4">
            <Link to="/blogs">
              <Button size="lg" className="bg-accent hover:bg-accent/90">
                Explore All Posts
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Featured Posts */}
      <section className="container mx-auto px-4 py-12 max-w-4xl">
        <div className="mb-8">
          <h2 className="text-3xl font-serif font-bold">Featured Articles</h2>
        </div>
        <div className="space-y-0">
          {featuredPosts.map((post) => (
            <BlogCard key={post.id} post={post} />
          ))}
        </div>
      </section>

      {/* Footer */}
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

export default Index;
