#pragma once

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <string>
#include <iomanip>
#include <vector>
#include <assert.h>


// vector print to output stream
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    std::cout << "[";
    for (int i = 0; i < vec.size(); ++i)
    {
        if (i > 0)
            os << ", ";
        os << vec[i];
    }
    std::cout << "]";
    return os;
}

// flattens vector of vectors into vector
namespace gv
{
template <typename T>
std::vector<T> flatten_vector(const std::vector<std::vector<T>>& vecvec)
{
    std::vector<T> vec;
    size_t size = vecvec.size();

    for (int i = 0; i < size; ++i)
        vec.insert(vec.end(), vecvec[i].begin(), vecvec[i].end());
    
    return vec;
}
}

/****************************************************************************************************/
// NATURAL LANGUAGE PROCESSING HELPER FUNCTIONS

namespace gv
{
namespace nlp
{

// returns string with all characters lower case
std::string str_lower(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(),
    [](unsigned char c){ return std::tolower(c); });
    return str;
}

// removes instances of a given char from given string
std::string remove_char(std::string str, const char& c)
{
    for (size_t i = str.find(c); i != str.npos; i = str.find(c, i))
        str.erase(i, 1);
    return str;
}

// removes any char that isn't a letter 65-90 (A, ..., Z) or 97-122 (a, ..., z)
// if template param == true then spaces are allowed
// exception set will allow chars that aren't in alphabet to remain in string
template <bool inc_space>
std::string only_alphabet(std::string str, const std::unordered_set<char>& except)
{
    for (size_t i = 0; i < str.size(); ++i)
        // check if char is not in alphabet, is not a space (provided spaces are allowed) and is not in exception set
        if ((str[i] < 65 || (str[i] > 90 && str[i] < 97) || str[i] > 122) 
            && (str[i] != (int)' ' || !inc_space) 
            && (except.find(str[i]) == except.end()))
        {
            str.erase(i, 1);
            --i;
        }
    return str;
}

// get all instances of str enclosed by left and right delimiters provided
// store instances in vector
std::vector<std::string> get_enclosed(const std::string& str, const char& lh, const char& rh)
{
    std::string buf;
    std::vector<std::string> sub;
    bool enc = false;

    for (size_t i = 0; i < str.size(); ++i)
    {
        if (str[i] == lh && enc == false)
        {
            enc = true;
            continue;
        }
        else if (str[i] == rh)
        {
            enc = false;
            sub.push_back(buf);
            buf = "";
        }
        
        if (enc == true)
            buf += str[i];        
    }
    return sub;
}

// removes string enclosed by left and right delimiters
// delimiters are also removed
std::string remove_enclosed(std::string str, const char& lh, const char& rh)
{
    bool enc = false;
    int start;

    for (size_t i = 0; i < str.size(); ++i)
    {
        if (str[i] == lh && enc == false)
        {
            enc = true;
            start = i;
        }
        else if (str[i] == rh)
        {
            enc = false;
            str.erase(start, i - start + 1);
        }
    }
    return str;
}

// returns sentences, i.e. strings that are delimited by a '.'
std::vector<std::string> get_sentences(const std::string& str)
{
    std::string buf;
    std::stringstream ss(str);
    std::vector<std::string> sentences;

    while (!ss.eof())
    {
        std::getline(ss, buf, '.');

        if (buf.size() > 0)
            sentences.push_back(buf);
    }

    return sentences;
}


// get words in a sentence string delimited by a space
std::vector<std::string> get_words(const std::string& str)
{
    std::string buf;
    std::stringstream ss(str);
    std::vector<std::string> words;

    while (!ss.eof())
    {
        std::getline(ss, buf, ' ');

        if (buf.size() > 0)
            words.push_back(buf);
    }

    return words;
}

// create ptr to unordered set of words in English dictionary
// delimiter is between each word in dictionary
// words are transformed and stored in lowercase
std::unordered_set<std::string>* get_dictionary_ptr(const std::string& path, const char& delim)
{
    std::ifstream is;
    std::string buf;
    is.open(path);

    std::unordered_set<std::string>* p_dict = new std::unordered_set<std::string>;

    while (!is.eof())
    {
        std::getline(is, buf, delim);
        p_dict->insert(str_lower(buf));
    }

    is.close();

    return p_dict;
}

// get lexicon vector of word strings
// words are transformed and stored in lower case
std::unordered_set<std::string>* get_lexicon_ptr(const std::vector<std::string>& words)
{
    std::unordered_set<std::string>* p_lex = new std::unordered_set<std::string>;

    for (auto& word : words)
        p_lex->insert(str_lower(word));

    return p_lex;
}

// returns vector of sentences that are enclosed in speech marks "..." and separated by '.'
// each sentence is represented as a vector of word strings
// each word string is stripped of punctuation including spaces and stored in lowercase
// punctation provided in exception list is not removed
// only sentences that have length >= min_len and only have words in dictionary provided are included
std::vector<std::vector<std::string>> get_speech_sentences(const std::string& path, const std::string& dict_path, 
    const int& min_len, const std::unordered_set<char>& punc_except)
{
    std::ifstream is;
    std::string buffer;
    std::string txt;

    is.open(path);

    // extract text into one long string without breaks
    if (is.is_open())
    {
        while (is.eof() == false)
        {
            std::getline(is, buffer, '\n');
            
            if (buffer.size() > 0)
                txt += buffer + " ";    
        }
    }

    // extract vector of all text within speech marks "..."
    std::vector<std::string> speech = gv::nlp::get_enclosed(txt, '"', '"');

    std::vector<std::vector<std::string>> sentences;

    // load English dictionary into ptr to unordered set
    std::unordered_set<std::string>* p_dict = gv::nlp::get_dictionary_ptr(dict_path, '\n');

    assert(p_dict != nullptr);

    for (int i = 0; i < speech.size(); ++i)
    {
        std::vector<std::string> buf = gv::nlp::get_sentences(speech[i]);
        for (int j = 0; j < buf.size(); ++j)
        {
            std::string buf2 = gv::nlp::only_alphabet<true>(buf[j], punc_except);
            std::vector<std::string> words = gv::nlp::get_words(buf2);

            bool all_english = true;
            for (int k = 0; k < words.size(); ++k)
                if (p_dict->find(gv::nlp::str_lower(words[k])) == p_dict->end())
                    all_english = false;

            // only include sentences with all words in English dictionary and at least minimum length
            if (words.size() > 8 && all_english == true)
                sentences.push_back(words);
        }
    }
    
    is.close();
    delete p_dict;
    return sentences;
}

// takes vector of sentences, where each sentence is represented by vector of word strings
// one-hot encodes against lexicon of words, represented by vector of word strings
// assumed that all word strings in sentences are present in lexicon
std::vector<std::vector<std::vector<double>>> to_onehot(const std::vector<std::vector<std::string>>& sentences, const std::vector<std::string>& lex)
{
    std::vector<std::vector<std::vector<double>>> oh(sentences.size());

    for (int i = 0; i < sentences.size(); ++i)
    {
        for (int j = 0; j < sentences[i].size(); ++j)
        {
            bool found = false;
            std::vector<double> v(lex.size(), 0.0);
            for (int k = 0; k < lex.size(); ++k)
            {
                if (lex[k] == str_lower(sentences[i][j]))
                {
                    found = true;
                    v[k] = 1.0;
                }
            }
            if (found == false)
            {
                std::cout << i << ", " << j << " " << sentences[i][j] << std::endl;
            }
            assert(found == true);
            oh[i].push_back(v);
        }
    }

    return oh;
}


} // namespace nlp
} // namespace gv