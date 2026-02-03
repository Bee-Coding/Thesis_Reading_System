"""
PDF解析器 - 提取论文文本内容和结构
"""
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """论文章节"""
    title: str
    content: str
    level: int = 1
    page_start: int = 0
    page_end: int = 0


@dataclass
class PaperContent:
    """论文内容"""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    sections: List[PaperSection] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    full_text: str = ""
    page_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_section(self, title_pattern: str) -> Optional[PaperSection]:
        """根据标题模式获取章节"""
        for section in self.sections:
            if re.search(title_pattern, section.title, re.IGNORECASE):
                return section
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "sections": [{"title": s.title, "content": s.content[:500] + "..."} for s in self.sections],
            "page_count": self.page_count,
            "full_text_length": len(self.full_text),
        }


class PDFParser:
    """PDF解析器"""
    
    # 主章节标题模式 (支持有空格和无空格两种格式)
    # 格式: "1. Introduction" 或 "1.Introduction" 或 "Introduction"
    SECTION_PATTERNS = [
        # 带编号的章节 (1. Introduction 或 1.Introduction)
        r'^(\d+)\.\s*(Introduction|INTRODUCTION)\b',
        r'^(\d+)\.\s*(Related\s*Work|RELATED\s*WORK|Background|BACKGROUND)\b',
        r'^(\d+)\.\s*(Method|METHOD|Methodology|METHODOLOGY|Approach|APPROACH|Methods|METHODS)\b',
        r'^(\d+)\.\s*(Experiment|EXPERIMENT|Experiments|EXPERIMENTS|Evaluation|EVALUATION|Results|RESULTS)\b',
        r'^(\d+)\.\s*(Discussion|DISCUSSION)\b',
        r'^(\d+)\.\s*(Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\b',
        r'^(\d+)\.\s*(Appendix|APPENDIX|Supplementary|SUPPLEMENTARY)\b',
        r'^(\d+)\.\s*(Ablation|ABLATION)\b',
        r'^(\d+)\.\s*(Implementation|IMPLEMENTATION)\b',
        # 不带编号的章节
        r'^(Abstract|ABSTRACT)\b',
        r'^(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\b',
        r'^(Acknowledgments?|ACKNOWLEDGMENTS?)\b',
    ]
    
    # 子章节模式 (2.1. xxx 或 2.1.xxx 或 2.1 xxx)
    SUBSECTION_PATTERNS = [
        r'^(\d+)\.(\d+)\.?\s*([A-Z][A-Za-z\s\-]+)',  # 2.1. End-to-End 或 2.1.End-to-End
    ]
    
    def __init__(self):
        self._pdfplumber = None
        self._pymupdf = None
    
    def parse(self, pdf_path: str) -> PaperContent:
        """解析PDF文件"""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Parsing PDF: {path.name}")
        
        # 尝试使用pdfplumber
        try:
            return self._parse_with_pdfplumber(path)
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyMuPDF")
        
        # 回退到PyMuPDF
        try:
            return self._parse_with_pymupdf(path)
        except Exception as e:
            logger.error(f"All PDF parsers failed: {e}")
            raise
    
    def _parse_with_pdfplumber(self, path: Path) -> PaperContent:
        """使用pdfplumber解析"""
        import pdfplumber
        
        content = PaperContent()
        full_text_parts = []
        
        with pdfplumber.open(path) as pdf:
            content.page_count = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                full_text_parts.append(text)
                
                # 从第一页提取标题和作者
                if i == 0:
                    self._extract_title_authors(text, content)
        
        content.full_text = "\n\n".join(full_text_parts)
        
        # 提取摘要
        self._extract_abstract(content)
        
        # 提取章节
        self._extract_sections(content)
        
        # 提取参考文献
        self._extract_references(content)
        
        logger.info(f"Parsed {content.page_count} pages, {len(content.sections)} sections")
        return content
    
    def _parse_with_pymupdf(self, path: Path) -> PaperContent:
        """使用PyMuPDF解析"""
        import fitz  # PyMuPDF
        
        content = PaperContent()
        full_text_parts = []
        
        doc = fitz.open(path)
        content.page_count = len(doc)
        
        for i, page in enumerate(doc):
            text = page.get_text()
            full_text_parts.append(text)
            
            if i == 0:
                self._extract_title_authors(text, content)
        
        doc.close()
        
        content.full_text = "\n\n".join(full_text_parts)
        self._extract_abstract(content)
        self._extract_sections(content)
        self._extract_references(content)
        
        return content
    
    def _extract_title_authors(self, first_page: str, content: PaperContent):
        """从第一页提取标题和作者"""
        lines = first_page.strip().split('\n')
        
        # 标题提取策略：
        # 1. 标题通常在前几行
        # 2. 排除arXiv标识、作者信息、机构信息
        # 3. 标题可能跨多行
        
        title_lines = []
        title_ended = False
        
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if not line:
                continue
            
            # 跳过arXiv标识
            if line.startswith('arXiv') or 'arxiv.org' in line.lower():
                continue
            
            # 检测是否到达作者/机构行（标题结束的标志）
            # 作者行通常包含：数字上标、逗号分隔的名字、机构关键词
            is_author_line = (
                re.search(r'[A-Z][a-z]+\d+,', line) or  # Name1,2
                re.search(r'\d+[A-Z][a-z]+\s+[A-Z][a-z]+', line) or  # 1John Smith
                'University' in line or 
                'Institute' in line or
                'Laboratory' in line or
                'Department' in line or
                '@' in line or
                re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s*,', line)  # John Smith,
            )
            
            if is_author_line:
                title_ended = True
                # 提取作者
                self._extract_authors_from_lines(lines[i:i+10], content)
                break
            
            # 跳过Abstract关键词（但不跳过包含Abstract的标题）
            if line == 'Abstract' or line == 'ABSTRACT':
                title_ended = True
                break
            
            if not title_ended:
                # 标题行通常不会太短，且不是纯数字
                if len(line) > 5 and not line.isdigit():
                    title_lines.append(line)
                    # 如果已经收集了足够长的标题，停止
                    if len(' '.join(title_lines)) > 100:
                        break
        
        # 合并标题行
        if title_lines:
            content.title = ' '.join(title_lines)
            # 清理标题中的多余空格
            content.title = re.sub(r'\s+', ' ', content.title).strip()
    
    def _extract_authors_from_lines(self, lines: List[str], content: PaperContent):
        """从多行中提取作者信息"""
        # 作者名字模式：FirstName LastName 或 F. LastName
        author_patterns = [
            r'([A-Z][a-z]+)\s+([A-Z][a-z]+)',  # John Smith
            r'([A-Z]\.)\s*([A-Z][a-z]+)',  # J. Smith
        ]
        
        seen_authors = set()
        for line in lines[:5]:
            line = line.strip()
            # 移除上标数字
            clean_line = re.sub(r'[\d∗†‡§¶\*]+', ' ', line)
            
            for pattern in author_patterns:
                matches = re.findall(pattern, clean_line)
                for match in matches:
                    author = f"{match[0]} {match[1]}"
                    if author not in seen_authors and len(author) > 3:
                        seen_authors.add(author)
                        content.authors.append(author)
                        if len(content.authors) >= 10:
                            return
    
    def _extract_abstract(self, content: PaperContent):
        """提取摘要"""
        text = content.full_text
        
        # 查找Abstract部分
        abstract_match = re.search(
            r'(?:Abstract|ABSTRACT)[:\s]*\n?(.*?)(?=\n\s*(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION|Keywords|Index Terms)|\nn)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # 清理多余空白
            abstract = re.sub(r'\s+', ' ', abstract)
            content.abstract = abstract[:2000]  # 限制长度
    
    def _extract_sections(self, content: PaperContent):
        """提取章节结构"""
        text = content.full_text
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        # 更灵活的章节检测模式
        # 支持: "1. Introduction", "1.Introduction", "Introduction", "1 Introduction"
        # 注意：有些PDF提取后没有空格，如 "2.RelatedWork"
        # 注意：双栏PDF可能导致章节标题出现在行中间
        main_section_keywords = (
            r'Introduction|Related\s*Work|RelatedWork|Background|'
            r'Method(?:s|ology)?|Approach|Experiment(?:s)?|Evaluation|Results|'
            r'Discussion|Conclusion(?:s)?|Appendix|Supplementary|'
            r'Ablation|Implementation|Preliminary|Overview|Problem|'
            r'Analysis|Limitation(?:s)?|Future\s*Work|FutureWork'
        )
        
        # 匹配行首的章节标题
        main_section_pattern_start = re.compile(
            rf'^(\d+)[\.\s]*({main_section_keywords})\b',
            re.IGNORECASE
        )
        
        # 匹配行中间的章节标题（双栏PDF情况）
        main_section_pattern_mid = re.compile(
            rf'\s(\d+)\.({main_section_keywords})\b',
            re.IGNORECASE
        )
        
        # 子章节模式: "2.1. xxx" 或 "2.1.xxx" 或 "2.1 xxx"
        subsection_pattern = re.compile(
            r'^(\d+)\.(\d+)[\.\s]*\s*([A-Z][A-Za-z\s\-&]+)',
            re.IGNORECASE
        )
        
        # 无编号的特殊章节
        special_section_pattern = re.compile(
            r'^(Abstract|References|Bibliography|Acknowledgments?)\s*$',
            re.IGNORECASE
        )
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_content.append('')
                continue
            
            new_section = None
            remaining_content = None
            
            # 检查是否是主章节标题（行首）
            main_match = main_section_pattern_start.match(line_stripped)
            if main_match:
                section_num = main_match.group(1)
                section_name = main_match.group(2)
                new_section = PaperSection(
                    title=f"{section_num}. {section_name}",
                    content="",
                    level=1
                )
            else:
                # 检查行中间是否有章节标题（双栏PDF情况）
                mid_match = main_section_pattern_mid.search(line_stripped)
                if mid_match:
                    section_num = mid_match.group(1)
                    section_name = mid_match.group(2)
                    new_section = PaperSection(
                        title=f"{section_num}. {section_name}",
                        content="",
                        level=1
                    )
                    # 保存章节标题之前的内容
                    remaining_content = line_stripped[:mid_match.start()].strip()
            
            # 检查子章节
            if not new_section:
                subsection_match = subsection_pattern.match(line_stripped)
                if subsection_match:
                    major = subsection_match.group(1)
                    minor = subsection_match.group(2)
                    name = subsection_match.group(3).strip()
                    if len(name) > 3 and len(name) < 80:
                        new_section = PaperSection(
                            title=f"{major}.{minor}. {name}",
                            content="",
                            level=2
                        )
            
            # 检查特殊章节
            if not new_section:
                special_match = special_section_pattern.match(line_stripped)
                if special_match:
                    new_section = PaperSection(
                        title=special_match.group(1),
                        content="",
                        level=1
                    )
            
            if new_section:
                # 如果有剩余内容，先添加到当前章节
                if remaining_content and current_section:
                    current_content.append(remaining_content)
                
                # 保存前一个章节
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    if len(current_section.content) > 50:
                        content.sections.append(current_section)
                
                current_section = new_section
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # 保存最后一个章节
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            if len(current_section.content) > 50:
                content.sections.append(current_section)
        
        # 按章节编号排序
        self._sort_sections(content)
    
    def _sort_sections(self, content: PaperContent):
        """按章节编号排序"""
        def get_section_key(section: PaperSection) -> tuple:
            """提取章节编号用于排序"""
            title = section.title
            # 匹配 "1. xxx" 或 "1.2. xxx" 格式
            match = re.match(r'^(\d+)(?:\.(\d+))?', title)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2)) if match.group(2) else 0
                return (major, minor)
            # 特殊章节放在最后
            if 'abstract' in title.lower():
                return (0, 0)
            if 'reference' in title.lower():
                return (999, 0)
            return (998, 0)
        
        content.sections.sort(key=get_section_key)
    
    def _extract_references(self, content: PaperContent):
        """提取参考文献"""
        text = content.full_text
        
        # 查找References部分 - 更灵活的匹配
        ref_start = -1
        for pattern in [r'\nReferences\b', r'\nREFERENCES\b', r'\nBibliography\b']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ref_start = match.start()
                break
        
        if ref_start < 0:
            return
        
        ref_text = text[ref_start:]
        
        # 提取所有 [数字] 格式的参考文献
        # 格式: [1] Author names. Title. Venue, year.
        ref_pattern = re.compile(r'\[(\d+)\]\s*([^[\]]+?)(?=\[\d+\]|$)', re.DOTALL)
        matches = ref_pattern.findall(ref_text)
        
        refs_dict = {}
        for num, content_text in matches:
            num = int(num)
            # 清理内容
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            # 移除可能混入的其他内容（如页码、图表说明等）
            content_text = re.sub(r'\s*\d+\s*$', '', content_text)  # 移除末尾的页码
            if len(content_text) > 20:  # 只保留有意义的内容
                refs_dict[num] = f"[{num}] {content_text[:500]}"
        
        # 按编号排序
        content.references = [refs_dict[k] for k in sorted(refs_dict.keys())]
