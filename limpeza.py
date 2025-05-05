from chatminer.chatparsers import WhatsAppParser
import pandas as pd

parser = WhatsAppParser('Conversa do WhatsApp com Uni.txt')
parser.parse_file()

df = parser.parsed_messages.get_df(as_pandas=True)

# removendo imagens e vídeos
df = df[~df['message'].str.contains('Mídia oculta')]
 # mensagens com link
df = df[~df['message'].str.startswith('https://')]

# removendo pessoas que falam pouco
df_cortado = df[~df['author'].isin(['Mariana Maia', 'Pedro'])].copy()

# selecionando 2.6k mensagens de cada membro
df_cortado = df_cortado.groupby("author").apply(lambda x: x.sample(n=2600, random_state=1)).reset_index(drop=True)

# salvando num arquivo pkl
df_cortado.to_pickle('whatsapp_unirio.pkl')