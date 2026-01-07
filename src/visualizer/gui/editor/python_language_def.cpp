/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_language_def.hpp"

namespace lfs::vis::editor {

    const TextEditor::LanguageDefinition& getPythonLanguageDef() {
        static bool initialized = false;
        static TextEditor::LanguageDefinition langDef;

        if (!initialized) {
            langDef.mName = "Python";
            langDef.mCaseSensitive = true;
            langDef.mAutoIndentation = true;
            langDef.mPreprocChar = '\0'; // Python has no preprocessor

            // Single line comment
            langDef.mSingleLineComment = "#";
            // Python doesn't have block comments like C, but we handle docstrings via regex
            langDef.mCommentStart = "";
            langDef.mCommentEnd = "";

            // Python keywords
            static const char* const keywords[] = {
                "False", "None", "True", "and", "as", "assert", "async", "await",
                "break", "class", "continue", "def", "del", "elif", "else", "except",
                "finally", "for", "from", "global", "if", "import", "in", "is",
                "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
                "try", "while", "with", "yield"};

            for (auto& k : keywords) {
                langDef.mKeywords.insert(k);
            }

            // Python built-in identifiers
            static const char* const builtins[] = {
                "abs", "aiter", "all", "anext", "any", "ascii", "bin", "bool",
                "breakpoint", "bytearray", "bytes", "callable", "chr", "classmethod",
                "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate",
                "eval", "exec", "filter", "float", "format", "frozenset", "getattr",
                "globals", "hasattr", "hash", "help", "hex", "id", "input", "int",
                "isinstance", "issubclass", "iter", "len", "list", "locals", "map",
                "max", "memoryview", "min", "next", "object", "oct", "open", "ord",
                "pow", "print", "property", "range", "repr", "reversed", "round",
                "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum",
                "super", "tuple", "type", "vars", "zip",
                // Common exceptions
                "Exception", "BaseException", "TypeError", "ValueError", "KeyError",
                "IndexError", "AttributeError", "RuntimeError", "StopIteration",
                "ImportError", "ModuleNotFoundError", "FileNotFoundError", "OSError",
                // Special attributes
                "__name__", "__doc__", "__class__", "__dict__", "__init__",
                "__new__", "__del__", "__repr__", "__str__", "__call__",
                "__iter__", "__next__", "__getitem__", "__setitem__", "__len__",
                "self", "cls"};

            for (auto& b : builtins) {
                TextEditor::Identifier id;
                id.mDeclaration = "Built-in";
                langDef.mIdentifiers.insert(std::make_pair(std::string(b), id));
            }

            // Token regex patterns
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "[ \\t]*#.*", TextEditor::PaletteIndex::Comment));

            // Triple-quoted strings (docstrings)
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    R"(\"\"\"[^\"]*\"\"\")", TextEditor::PaletteIndex::MultiLineComment));
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    R"(\'\'\'[^\']*\'\'\')", TextEditor::PaletteIndex::MultiLineComment));

            // String literals (with raw/f-string prefixes)
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    R"([rRfFbBuU]?\"[^\"\\]*(\\.[^\"\\]*)*\")", TextEditor::PaletteIndex::String));
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    R"([rRfFbBuU]?\'[^\'\\]*(\\.[^\'\\]*)*\')", TextEditor::PaletteIndex::String));

            // Numbers: hex, binary, octal, float with exponent, int
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "0[xX][0-9a-fA-F]+[uUlL]*", TextEditor::PaletteIndex::Number));
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "0[bB][01]+", TextEditor::PaletteIndex::Number));
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "0[oO][0-7]+", TextEditor::PaletteIndex::Number));
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "[0-9]+[eE][+-]?[0-9]+[jJ]?", TextEditor::PaletteIndex::Number));
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "[0-9]+(\\.[0-9]*)?([eE][+-]?[0-9]+)?[jJ]?", TextEditor::PaletteIndex::Number));
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "\\.[0-9]+([eE][+-]?[0-9]+)?[jJ]?", TextEditor::PaletteIndex::Number));

            // Decorators
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "@[a-zA-Z_][a-zA-Z0-9_.]*", TextEditor::PaletteIndex::Preprocessor));

            // Identifiers
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "[a-zA-Z_][a-zA-Z0-9_]*", TextEditor::PaletteIndex::Identifier));

            // Punctuation
            langDef.mTokenRegexStrings.push_back(
                std::make_pair<std::string, TextEditor::PaletteIndex>(
                    "[\\[\\]\\{\\}\\(\\)\\<\\>\\=\\+\\-\\*\\/\\%\\^\\&\\|\\~\\!\\?\\:\\;\\,\\.]",
                    TextEditor::PaletteIndex::Punctuation));

            initialized = true;
        }

        return langDef;
    }

} // namespace lfs::vis::editor
