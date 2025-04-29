import { useState, useRef, useEffect, Fragment, useCallback } from "react";
import { getCodeString } from "rehype-rewrite";
import mermaid from "mermaid";
import panzoom from "panzoom";

const randomid = () => parseInt(String(Math.random() * 1e15), 10).toString(36);

export const MarkdownCode = ({ inline, children = [], className, ...props }: any) => {
  const demoid = useRef(`dome${randomid()}`);
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  const isMermaid = className && /^language-mermaid/.test(className.toLocaleLowerCase());
  const code = props.node && props.node.children ? getCodeString(props.node.children) : children[0] || '';
  const panzoomInstance = useRef<any>(null);

  const reRender = async () => {
    if (container && isMermaid) {
      try {
        const str = await mermaid.render(demoid.current, code);
        container.innerHTML = str.svg;
        // Enable panzoom on the SVG
        const svg = container.querySelector('svg');
        if (svg) {
          svg.style.width = '100%';
          svg.style.maxWidth = '600px';
          svg.style.display = 'block';
          svg.style.margin = '0 auto';
          if (panzoomInstance.current) {
            panzoomInstance.current.dispose();
          }
          panzoomInstance.current = panzoom(svg, {
            maxZoom: 8,
            minZoom: 0.2,
            bounds: true,
            boundsPadding: 0.2,
            zoomDoubleClickSpeed: 1,
          });
        }
      } catch (error: any) {
        container.innerHTML = error?.message || String(error);
      }
    }
  };

  useEffect(() => {
    reRender();
    return () => {
      if (panzoomInstance.current) {
        panzoomInstance.current.dispose();
        panzoomInstance.current = null;
      }
    };
    // eslint-disable-next-line
  }, [container, isMermaid, code, demoid]);

  const refElement = useCallback((node: HTMLDivElement | null) => {
    if (node !== null) {
      setContainer(node);
    }
  }, []);

  if (isMermaid) {
    return (
      <Fragment>
        <code id={demoid.current} style={{ display: "none" }} />
        <div
          ref={refElement}
          data-name="mermaid"
          style={{
            width: '100%',
            minHeight: 120,
            height: 600,
            maxWidth: 700,
            margin: '0 auto',
            overflow: 'auto',
            border: '1px solid #e5e7eb',
            borderRadius: 8,
            background: '#fff',
            cursor: 'grab',
          }}
        />
      </Fragment>
    );
  }
  return <code className={className}>{children}</code>;
};
