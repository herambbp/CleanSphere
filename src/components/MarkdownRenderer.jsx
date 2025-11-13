import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';

export default function MarkdownRenderer({ content }) {
  return (
    <div className="prose prose-lg max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight, rehypeRaw]}
        components={{
          h1: ({ node, ...props }) => (
            <h1 className="text-4xl font-bold text-gray-900 mt-8 mb-4" {...props} />
          ),
          h2: ({ node, children, ...props }) => {
            const id = children.toString().toLowerCase().replace(/[^\w]+/g, '-');
            return (
              <h2
                id={id}
                className="text-3xl font-bold text-gray-900 mt-8 mb-4 scroll-mt-20"
                {...props}
              >
                {children}
              </h2>
            );
          },
          h3: ({ node, children, ...props }) => {
            const id = children.toString().toLowerCase().replace(/[^\w]+/g, '-');
            return (
              <h3
                id={id}
                className="text-2xl font-semibold text-gray-800 mt-6 mb-3 scroll-mt-20"
                {...props}
              >
                {children}
              </h3>
            );
          },
          p: ({ node, ...props }) => (
            <p className="text-gray-700 leading-relaxed mb-4" {...props} />
          ),
          code: ({ node, inline, className, children, ...props }) => {
            return inline ? (
              <code
                className="bg-gray-100 text-pink-600 px-1.5 py-0.5 rounded text-sm font-mono"
                {...props}
              >
                {children}
              </code>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          pre: ({ node, ...props }) => (
            <pre className="bg-gray-900 text-gray-100 rounded-xl p-4 overflow-x-auto my-6" {...props} />
          ),
          ul: ({ node, ...props }) => (
            <ul className="list-disc list-inside space-y-2 mb-4 text-gray-700" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="list-decimal list-inside space-y-2 mb-4 text-gray-700" {...props} />
          ),
          blockquote: ({ node, ...props }) => (
            <blockquote
              className="border-l-4 border-primary-500 pl-4 italic text-gray-700 my-4"
              {...props}
            />
          ),
          a: ({ node, ...props }) => (
            <a className="text-primary-600 hover:text-primary-700 underline" {...props} />
          ),
          img: ({ node, ...props }) => (
            <img className="rounded-lg shadow-md my-6 max-w-full" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
