import { Link } from "react-router-dom";

const Navigation = () => {
  return (
    <nav className="border-b border-border bg-background sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 max-w-5xl">
        <div className="flex items-center justify-between">
          <Link to="/" className="text-2xl font-serif font-bold hover:text-accent transition-colors">
            ML & CS Insights
          </Link>
          <div className="flex gap-8">
            <Link to="/" className="text-foreground hover:text-accent transition-colors font-medium">
              Home
            </Link>
            <Link to="/blogs" className="text-foreground hover:text-accent transition-colors font-medium">
              All Posts
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
