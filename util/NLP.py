from spacy.lang import *
import re

class NLP:
    french_vowels = ['a','e','i','o','u','h','î', 'y','œ','é','à', 'è', 'ù','â', 'ê', 'î', 'ô', 'û','ë', 'ï', 'ü']
    french_vowels_replacement = {'à':'a','â':'a','é':'e','è':'e','ê':'e','ë':'e','î':'i','ï':'i','ô':'o','œ':'oe', 'ù':'u', 'û':'u', 'ü':'u'}

    @staticmethod
    def replaceSpecialVowels(text):
        for vowel in [vowel for vowel in NLP.french_vowels_replacement if vowel in text]:
            text = text.replace(vowel, NLP.french_vowels_replacement[vowel])
        return text

    @staticmethod
    def checkIfStartsWithEnumerator(text, lang='fr', includePrefixWord=None, replaceRegexForLine=False):
        if includePrefixWord is None:
            word = ''
        else:
            if includePrefixWord == 'DefaultNotes':
                includePrefixWord = {'fra': ['note'], 'eng': ['note'], 'deu': ['punkt', 'erläuterung','anmerkung']}

            if len(includePrefixWord[lang])==1:
                word = '(?:' + includePrefixWord[lang][0] + '\s)?'
            else:
                word = '(?:'
                for wordFilter in includePrefixWord[lang]:
                    word += wordFilter + "|"
                word +='\s)*'
        if not replaceRegexForLine:
            enumeratorRegex1 = '^[•>✓.\-–—_\-\*�]?' + word + '[1livx]?[0-9a-j]?[.]?[0-9]{0,2}[1livx]*[.]?[0-9]?\s?[).\-–_:\-—\*]{0,2}\s'
        else:
            enumeratorRegex1 = '^[•>✓.\-–—_\-\*�]?' + word + '[1livx]?[0-9a-j]?[.]?[0-9]{0,2}[1livx]*[.]?[0-9]?\s?[).\-–_:\-—\*]{1,2}\s'

        enumeratorRegex2 = '^[•>✓.\-–—_\\*�]?' + word + '[0-9]{1,2}[.]?[0-9]{0,2}[.]?[0-9]{0,2}[).\-–—_:\-\*\s]{1,2}'
        enumeratorRegex3 = '^[•>✓.\-–—_\-\*�]\s?' + word
        enumeratorRegex4 = '^([0-9a-f][.-][0-9a-f]?[.-]?)+'
        results3 = re.findall(enumeratorRegex3, text.lower().strip())
        results1 = re.findall(enumeratorRegex1, text.lower().strip())
        results2 = re.findall(enumeratorRegex2, text.lower().strip())
        results4 = re.findall(enumeratorRegex4, text.lower().strip())

        results1.extend(results2)
        results1.extend(results3)
        results1.extend(results4)
        if len(text) >0:
            first_char = text[0]
            if ord(first_char) in [65533, 61623]:
                results1.extend(first_char)

        return results1

    @staticmethod
    def hasText(text):
        textRegex = '[a-z\'\-]+'
        results = re.findall(textRegex, text.lower())
        if len (results)>0:
            return True
        return False

    @staticmethod
    def getTypeEnumeratorPattern(enumerator):
        text = enumerator.lower().strip()
        enumeratorRegex1 = '[ivxl]+'
        enumeratorRegex2 = '[0-9]+'
        enumeratorRegex3 = '[a-z]*'
        results1 = re.findall(enumeratorRegex1, text)
        for result in results1:
            text = text.replace(result," romain ")
        results2 = re.findall(enumeratorRegex2, text)
        for result in results2:
            text = text.replace(result," cardinal ")
        results3 = re.findall(enumeratorRegex3, text)
        if len(results3[0]) == 1:
            text = text.replace(results3[0]," lettre ")
        text = NLP.replaceSpecialCaractersWithText(text)
        return text

    @staticmethod
    def removeDatesInSentence(subTitlesList):
        patternYear4Digits = "(\s[0-3]?[0-9][-/.\s]\s*(?:[a-zÀ-Ÿ']*([0-3]?[0-9])?){1}[-/.\s]\s*[12][089][0-9]{2}\s)"
        patternYear2Digits = "(\s[0-3]?[0-9][-/.\s]\s*(?:[a-zÀ-Ÿ']*([0-3]?[0-9])?){1}[-/.\s]\s*[0-9]{1,2}\s)"
        newList = []
        for text in subTitlesList:
            coincidences = re.findall(patternYear4Digits, ' ' + text + ' ')
            if len(coincidences) > 0:
                coincidences = [t for c in coincidences for t in c if len(t) >0 ]
                text = text.replace(coincidences[0].strip(),'').strip()
            coincidences = re.findall(patternYear2Digits, ' ' + text + ' ')
            if len(coincidences) > 0:
                coincidences = [t for c in coincidences for t in c if len(t) >0 ]
                text = text.replace(coincidences[0].strip(), '').strip()
            newList.append(text)
        return newList

    @staticmethod
    def removeNumbersInText(text):
        if type(text) == list: text = text[0]
        text = NLP.findNumbersInText(text, False, False, replace=True,replacement_text="")
        return text


    @staticmethod
    def replaceSpecialCaractersWithText(text):
        text = text.replace("-", " petit tiret ")
        text = text.replace("–", " tiret ")
        text = text.replace("_", " souligner ")
        text = text.replace(".", " point ")
        text = text.replace("✓", " coche ")
        text = text.replace("*", " astérisque ")
        text = text.replace(")", " parenthèse ")
        return text

    @staticmethod
    def findNumbersInText(text, beginningOfLine=False, endOfLine=False,replace=False, replacement_text=""):
        start = ''
        end = ''
        if beginningOfLine:
            start = '^'
        if endOfLine:
            end = '$'
        regexNumberPattern = start + '(\s[a-z]*[+-]?\s?\d{0,3}[,.\s]*\d{0,3}[,.\s]*[a-z]?\d{2,3}[,.\s]?\d*[€|$]*(\beuros\b)*(\beur\b)*[.,]?\s)' + end
        regexNumberPattern2 = start + '(\s\d[,.\s]\d\s?\d[€|$]*(\beuros\b)*(\beur\b)*[.,]?\s)' + end 
        if not replace:
            coincidencesNumbers = re.findall(regexNumberPattern," " + text + " ")
            coincidencesNumbers2 = re.findall(regexNumberPattern2 ," " + text + " ") 
            coincidencesNumbers.extend(coincidencesNumbers2) 
            return [item[0] for item in coincidencesNumbers]
        else:
            text = re.sub(regexNumberPattern,replacement_text," " + text + " ")
            text = re.sub(regexNumberPattern2,replacement_text," " + text + " ")
            return text.strip()