# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile

import pytest
from jsonschema import ValidationError

from rasa_nlu import training_data
from rasa_nlu import utils
from rasa_nlu.convert import convert_training_data
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa_nlu.training_data.formats.rasa import validate_rasa_nlu_data


def test_example_training_data_is_valid():
    demo_json = 'data/examples/rasa/demo-rasa.json'
    data = utils.read_json_file(demo_json)
    validate_rasa_nlu_data(data)


@pytest.mark.parametrize("invalid_data", [
    {"wrong_top_level": []},
    ["this is not a toplevel dict"],
    {"rasa_nlu_data": {
        "common_examples": [{
            "intent": "some example without text"}]}},
    {"rasa_nlu_data": {
        "common_examples": [{
            "text": "mytext", "entities": [
                {"start": "INVALID", "end": 0, "entity": "x"}]
        }]
    }},
])
def test_validation_is_throwing_exceptions(invalid_data):
    with pytest.raises(ValidationError):
        validate_rasa_nlu_data(invalid_data)


def test_luis_data():
    td = training_data.load_data('data/examples/luis/demo-restaurants.json')
    assert len(td.entity_examples) == 8
    assert len(td.intent_examples) == 28
    assert len(td.training_examples) == 28
    assert td.entity_synonyms == {}
    assert td.intents == {"affirm", "goodbye", "greet", "inform"}
    assert td.entities == {"location", "cuisine"}


def test_wit_data():
    td = training_data.load_data('data/examples/wit/demo-flights.json')
    assert len(td.entity_examples) == 4
    assert len(td.intent_examples) == 1
    assert len(td.training_examples) == 4
    assert td.entity_synonyms == {}
    assert td.intents == {"flight_booking"}
    assert td.entities == {"location", "datetime"}


def test_dialogflow_data():
    td = training_data.load_data('data/examples/dialogflow/')
    assert len(td.entity_examples) == 5
    assert len(td.intent_examples) == 24
    assert len(td.training_examples) == 24
    assert td.intents == {"affirm", "goodbye", "hi", "inform"}
    assert td.entities == {"cuisine", "location"}
    non_trivial_synonyms = {k: v for k, v in td.entity_synonyms.items() if k != v}
    assert non_trivial_synonyms == {"mexico": "mexican",
                                    "china": "chinese",
                                    "india": "indian"}


@pytest.mark.parametrize("filename", ["data/examples/rasa/demo-rasa.json", 'data/examples/rasa/demo-rasa.md'])
def test_demo_data(filename):
    td = training_data.load_data(filename)
    assert td.intents == {"affirm", "greet", "restaurant_search", "goodbye"}
    assert td.entities == {"location", "cuisine"}
    assert len(td.training_examples) == 42
    assert len(td.intent_examples) == 42
    assert len(td.entity_examples) == 11

    assert td.entity_synonyms == {'Chines': 'chinese',
                                  'Chinese': 'chinese',
                                  'chines': 'chinese',
                                  'vegg': 'vegetarian',
                                  'veggie': 'vegetarian'}

    assert td.regex_features == [{"name": "greet", "pattern": "hey[^\s]*"},
                                 {"name": "zipcode", "pattern": "[0-9]{5}"}]


@pytest.mark.parametrize("files", [('data/examples/rasa/demo-rasa.json', 'data/test/multiple_files_json'),
                                   ('data/examples/rasa/demo-rasa.md', 'data/test/multiple_files_markdown')])
def test_data_merging(files):
    td_reference = training_data.load_data(files[0])
    td = training_data.load_data(files[1])
    assert len(td.entity_examples) == len(td_reference.entity_examples)
    assert len(td.intent_examples) == len(td_reference.intent_examples)
    assert len(td.training_examples) == len(td_reference.training_examples)
    assert td.intents == td_reference.intents
    assert td.entities == td_reference.entities
    assert td.entity_synonyms == td_reference.entity_synonyms
    assert td.regex_features == td_reference.regex_features


def test_markdown_single_sections():
    td_regex_only = training_data.load_data('data/test/markdown_single_sections/regex_only.md')
    assert td_regex_only.regex_features == [{"name": "greet", "pattern": "hey[^\s]*"}]

    td_syn_only = training_data.load_data('data/test/markdown_single_sections/synonyms_only.md')
    assert td_syn_only.entity_synonyms == {'Chines': 'chinese',
                                           'Chinese': 'chinese'}


def test_repeated_entities():
    data = """
{
  "rasa_nlu_data": {
    "common_examples" : [
      {
        "text": "book a table today from 3 to 6 for 3 people",
        "intent": "unk",
        "entities": [
          {
            "entity": "description",
            "start": 35,
            "end": 36,
            "value": "3"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = training_data.load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example.get("entities")
        assert len(entities) == 1
        tokens = WhitespaceTokenizer().tokenize(example.text)
        start, end = MitieEntityExtractor.find_entity(entities[0],
                                                      example.text,
                                                      tokens)
        assert start == 9
        assert end == 10


def test_multiword_entities():
    data = """
{
  "rasa_nlu_data": {
    "common_examples" : [
      {
        "text": "show me flights to New York City",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 32,
            "value": "New York City"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = training_data.load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example.get("entities")
        assert len(entities) == 1
        tokens = WhitespaceTokenizer().tokenize(example.text)
        start, end = MitieEntityExtractor.find_entity(entities[0],
                                                      example.text,
                                                      tokens)
        assert start == 4
        assert end == 7


def test_nonascii_entities():
    data = """
{
  "luis_schema_version": "2.0",
  "utterances" : [
    {
      "text": "I am looking for a ßäæ ?€ö) item",
      "intent": "unk",
      "entities": [
        {
          "entity": "description",
          "startPos": 19,
          "endPos": 26
        }
      ]
    }
  ]
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = training_data.load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example.get("entities")
        assert len(entities) == 1
        entity = entities[0]
        assert entity["value"] == "ßäæ ?€ö)"
        assert entity["start"] == 19
        assert entity["end"] == 27
        assert entity["entity"] == "description"


def test_entities_synonyms():
    data = u"""
{
  "rasa_nlu_data": {
    "entity_synonyms": [
      {
        "value": "nyc",
        "synonyms": ["New York City", "nyc", "the big apple"]
      }
    ],
    "common_examples" : [
      {
        "text": "show me flights to New York City",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 32,
            "value": "NYC"
          }
        ]
      },
      {
        "text": "show me flights to nyc",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 22,
            "value": "nyc"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = training_data.load_data(f.name)
        assert td.entity_synonyms["New York City"] == "nyc"


def cmp_message_list(firsts, seconds):
    assert len(firsts) == len(seconds), "Message lists have unequal length"


def cmp_dict_list(firsts, seconds):
    if len(firsts) != len(seconds):
        return False

    for a in firsts:
        for idx, b in enumerate(seconds):
            if hash(a) == hash(b):
                del seconds[idx]
                break
        else:
            others = ", ".join([e.text for e in seconds])
            assert False, "Failed to find message {} in {}".format(a.text,
                                                                   others)
    return not seconds


@pytest.mark.parametrize("data_file,gold_standard_file,output_format,language",
                         [
                             ("data/examples/wit/demo-flights.json",
                              "data/test/wit_converted_to_rasa.json", "json",
                              None),
                             ("data/examples/luis/demo-restaurants.json",
                              "data/test/luis_converted_to_rasa.json", "json",
                              None),
                             ("data/examples/dialogflow/",
                              "data/test/dialogflow_en_converted_to_rasa.json",
                              "json", "en"),
                             ("data/examples/dialogflow/",
                              "data/test/dialogflow_es_converted_to_rasa.json",
                              "json", "es"),
                             ("data/examples/rasa/demo-rasa.md",
                              "data/test/md_converted_to_json.json", "json",
                              None),
                             ("data/examples/rasa/demo-rasa.json",
                              "data/test/json_converted_to_md.md", "md", None)
                         ])
def test_training_data_conversion(tmpdir, data_file, gold_standard_file,
                                  output_format, language):
    out_path = tmpdir.join("rasa_nlu_data.json")
    convert_training_data(data_file, out_path.strpath, output_format, language)
    td = training_data.load_data(out_path.strpath, language)
    assert td.entity_examples != []
    assert td.intent_examples != []

    gold_standard = training_data.load_data(gold_standard_file, language)
    cmp_message_list(td.entity_examples, gold_standard.entity_examples)
    cmp_message_list(td.intent_examples, gold_standard.intent_examples)
    assert td.entity_synonyms == gold_standard.entity_synonyms

    # converting the converted file back to original
    # file format and performing the same tests
    rto_path = tmpdir.join("data_in_original_format.txt")
    convert_training_data(out_path.strpath, rto_path.strpath, 'json', language)
    rto = training_data.load_data(rto_path.strpath, language)
    cmp_message_list(gold_standard.entity_examples, rto.entity_examples)
    cmp_message_list(gold_standard.intent_examples, rto.intent_examples)
    assert gold_standard.entity_synonyms == rto.entity_synonyms

    # If the above assert fails - this can be used
    # to dump to the file and diff using git
    # with io.open(gold_standard_file) as f:
    #     f.write(td.as_json(indent=2))
