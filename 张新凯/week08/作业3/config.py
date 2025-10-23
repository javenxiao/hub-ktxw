import codecs
import configparser

DOMAINS = codecs.open('assets/domains.txt', encoding="utf-8").readlines()
DOMAINS = [x.strip() for x in DOMAINS]

INTENTS = codecs.open('assets/intents.txt', encoding="utf-8").readlines()
INTENTS = [x.strip() for x in INTENTS]

SLOTS = codecs.open('assets/slots.txt', encoding="utf-8").readlines()
SLOTS = [x.strip() for x in SLOTS]

config = configparser.ConfigParser()
config.read('assets/config.ini')

API_KEY = config.get('llm', 'api_key')
BASE_URL = config.get('llm', 'base_url')
MODEL = config.get('llm', 'model')
