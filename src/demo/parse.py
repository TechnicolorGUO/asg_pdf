import os
import re
import json
import spacy
# from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader

# load spaCy model
nlp = spacy.load("en_core_web_sm")

class DocumentLoading:
    
    def extract_names(self, text):
        doc = nlp(text)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        return list(set(names))
    
    def clean_authors_text(self, text):
        # clear emails
        cleaned_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # clear academic keywords
        academic_keywords = [
            r'\bInstitute\b', r'\bUniversity\b', r'\bDepartment\b', r'\bCollege\b', r'\bSchool\b', r'\bFaculty\b', r'\bLaboratory\b',
            r'\bResearch\b', r'\bCenter\b', r'\bCentre\b', r'\bProgram\b', r'\bProgramme\b', r'\bGroup\b', r'\bDivision\b',
            r'\bUnit\b', r'\bCampus\b', r'\bBuilding\b', r'\bRoom\b', r'\bStreet\b', r'\bAvenue\b', r'\bBoulevard\b', r'\bRoad\b',
            r'\bCity\b', r'\bTown\b', r'\bState\b', r'\bProvince\b', r'\bCountry\b', r'\bZip\b', r'\bCode\b', r'\bPostal\b',
            r'\bBox\b', r'\bPhone\b', r'\bFax\b', r'\bEmail\b', r'\bWeb\b', r'\bSite\b', r'\bURL\b', r'\bAddress\b', r'\bContact\b',
            r'\bFloor\b', r'\bSuite\b', r'\bArea\b', r'\bDistrict\b', r'\bBlock\b'
        ]
        cleaned_text = re.sub('|'.join(academic_keywords), '', cleaned_text, flags=re.IGNORECASE)

        # clear extra spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text
    
    def extract_information_unstructured(self, text):
        data = {
            'title': "NONE",
            'authors': "NONE",
            'abstract': "NONE",
            'introduction': "NONE"
        }

        title_match = re.search(r'^(.*?)(?:\n|$)', text)
        if title_match:
            data['title'] = title_match.group(1).strip()
        
        authors_match = re.search(r'\n\n(.*?)(?:\n\nABSTRACT|\n\nAbstract|\n\nAbstract\.|\n\nAbstract—)', text, re.DOTALL)
        if authors_match:
            authors_text = authors_match.group(1).replace('\n', ' ').strip()
            cleaned_authors_text = self.clean_authors_text(authors_text)
            authors = self.extract_names(cleaned_authors_text)
            authors = [re.sub(r'\d+$', '', author).strip() for author in authors]  # clear trailing numbers
            data['authors'] = ", ".join(authors)

        abstract_match = re.search(
            r'(?i)ABSTRACT\s*(.*?)(?=\n\s*1\.?\s*INTRODUCTION|\n\s*1\.?\s*Introduction|\n\s*1\.?\s*INTRODUCTION|\n\s*1\.?\s*Introduction|\n\s*1\n\nIntroduction|\n\s*1\.\n\nIntroduction|\n\s*1\n\nINTRODUCTION|\n\s*1\.\n\nINTRODUCTION)',
            text, re.DOTALL
        )        
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            # clear leading dot
            if abstract_text.startswith('.'):
                abstract_text = abstract_text[1:].strip()
            data['abstract'] = abstract_text
        
        intro_match = re.search(r'(?i)(?:\d\.\s*INTRODUCTION|INTRODUCTION)\s*(.*?)(?=\n\n\d\.\s*\w+|\n\n\w+\n|$)', text, re.DOTALL)
        if intro_match:
            data['introduction'] = intro_match.group(1).strip()
        return data

    def extract_information_pypdf(self, text):
        data = {
            'title': "NONE",
            'authors': "NONE",
            'abstract': "NONE",
            'introduction': "NONE"
        }

        # assume title is the first 2 lines
        title_match = re.search(r'^(.*?)\n(.*?)\n', text, re.DOTALL)
        if title_match:
            data['title'] = title_match.group(1).strip() + ' ' + title_match.group(2).strip()

        authors_match = re.search(r'\n(.+?)\n(?:\n|ABSTRACT|INTRODUCTION)', text, re.DOTALL)
        # authors_match = re.search(rf'{re.escape(data["title"])}\n(.+?)(?:\nABSTRACT|\nINTRODUCTION)', text, re.DOTALL)
        # authors_match = re.search(rf'{re.escape(data["title"])}(.*?)\n(?:ABSTRACT|Abstract|Abstract\.|Abstract—)', text, re.DOTALL)

        if authors_match:
            data['authors'] = authors_match.group(1).replace('\n', ' ').strip()
            print(data['authors'])
            # authors_text = authors_match.group(1).replace('\n', ' ').strip()
            # cleaned_authors_text = self.clean_authors_text(authors_text)
            # authors = self.extract_names(cleaned_authors_text)
            # authors = [re.sub(r'\d+$', '', author).strip() for author in authors]
            # data['authors'] = ", ".join(authors)
        else:
            print("Authors not found")

        # authors_match = re.search(r'\n(.*?)\n(?:ABSTRACT|Abstract|Abstract\.|Abstract—)', text, re.DOTALL)
        # if authors_match:
        #     data['title'] = authors_match.group(1).strip()

        abstract_match = re.search(
            r'(?i)ABSTRACT\s*(.*?)(?=\n\s*\d+\.\s*INTRODUCTION|\n\s*\d+\.\s*Introduction|\n\s*\d+\sINTRODUCTION|\n\s*\d+\sIntroduction)',
            text, re.DOTALL
        )
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
            if abstract_text.startswith('.'):
                abstract_text = abstract_text[1:].strip()
            data['abstract'] = abstract_text

        intro_match = re.search(r'(?i)(?:\d+\.\s*INTRODUCTION|\d+\s*INTRODUCTION|INTRODUCTION)\s*(.*?)(?=\n\s*\d+\.\s*\w+|\n\s*\w+\n|$)', text, re.DOTALL)
        if intro_match:
            data['introduction'] = intro_match.group(1).strip()
        return data

    def clean_headers_footers(self, text_blocks):
        block_count = {}
        for block in text_blocks:
            if block in block_count:
                block_count[block] += 1
            else:
                block_count[block] = 1

        # clear blocks that appear in every page (likely headers/footers)
        header_footer_blocks = {block for block, count in block_count.items() if count > 1}
        cleaned_blocks = [block for block in text_blocks if block not in header_footer_blocks]
        return cleaned_blocks

    # 处理页边 页面注释等 枚举
    def clean_annotations(self, text_blocks):
        cleaned_blocks = []
        for block in text_blocks:
            # clear blocks that are likely annotations
            if re.search(r'\d{4}\s\d+\s\w+\sConference\s.*?\|\s.*?\|\sDOI:.*?\s\|\s\w+:\s.*?\n', block, flags=re.DOTALL) or \
            re.search(r'http\S+', block) or \
            re.search(r'\d+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+', block, flags=re.DOTALL):
                continue
            cleaned_blocks.append(block)
        return cleaned_blocks

    def clean_hyphenation_unstructured(self, text):
        # remove hyphenation across lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # remove hyphenation within lines
        text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)        
        return text
    
    def clean_hyphenation_pypdf(self, text):
        # remove hyphenation across lines, ensuring no space between words
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text

    def clean_empty_and_numeric_lines(self, text):
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip() or line.strip().isdigit():
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)
    
    def remove_newlines(self, text):
        return text.replace('\n', ' ')

    def unstructured_loader(self, file_path):
        loader = UnstructuredPDFLoader(file_path)
        documents = loader.load()
        # print(documents[0]) # for testing
        text_blocks = [doc.page_content for doc in documents]

        # Split each page into blocks using double newlines as the delimiter
        split_blocks = [block.split('\n\n') for block in text_blocks]
        flat_blocks = [item for sublist in split_blocks for item in sublist]

        cleaned_blocks = self.clean_headers_footers(flat_blocks)
        cleaned_blocks = self.clean_annotations(cleaned_blocks)
        text = "\n\n".join(cleaned_blocks)
        text = self.clean_hyphenation_unstructured(text)
        extracted_data = self.extract_information_unstructured(text)

        title = extracted_data['title']
        title_new = title.strip()
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            title_new = title_new.replace(char, '_')
        # print("============================")
        # print(title_new)

        print(os.getcwd())
        os.makedirs('./src/static/data/txt/', exist_ok=True)
        with open(f'./src/static/data/txt/{title_new}.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        print(extracted_data)
        return title

    # single \n between sections
    def pypdf_loader(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        # print(documents) # for testing
        # print(documents[0]) # for testing

        text_blocks = [doc.page_content for doc in documents]
        split_blocks = [block.split('\n') for block in text_blocks]
        flat_blocks = [item for sublist in split_blocks for item in sublist]
        cleaned_blocks = self.clean_headers_footers(flat_blocks)
        # cleaned_blocks = self.clean_annotations(cleaned_blocks)
        text = "\n".join(cleaned_blocks)
        # combine all pages into one text
        text = "\n".join([doc.page_content for doc in documents])
        # clean empty lines and numeric lines
        text = self.clean_empty_and_numeric_lines(text)
        # print(text)
        extracted_data = self.extract_information_pypdf(text)
        extracted_data = {k: self.clean_hyphenation_pypdf(v) for k, v in extracted_data.items()}
        extracted_data = {k: self.remove_newlines(v) for k, v in extracted_data.items()}
        title = extracted_data['title']

        title_new = title.strip()
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*',' ']
        for char in invalid_chars:
            title_new = title_new.replace(char, '_')
        # print("============================")
        # print(title_new)
        os.makedirs('./src/static/data/txt/', exist_ok=True)
        print(os.getcwd())
        with open(f'./src/static/data/txt/{title_new}.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        print(extracted_data)
        return title

if __name__ == "__main__":
    parser = DocumentLoading()
    # parser.unstructured_loader("./Test1.pdf")
    # parser.unstructured_loader("./Test3.pdf")
    # parser.unstructured_loader("./Test5.pdf")
    # parser.unstructured_loader("./Test6.pdf")
    # parser.unstructured_loader("./Test7.pdf")
    # parser.unstructured_loader("./Test8.pdf")

    parser.pypdf_loader("./Test1.pdf")
    # parser.pypdf_loader("./Test3.pdf")
    # parser.pypdf_loader("./Test5.pdf")
    # parser.pypdf_loader("./Test6.pdf")
    # parser.pypdf_loader("./Test7.pdf")
    # parser.pypdf_loader("./Test8.pdf")