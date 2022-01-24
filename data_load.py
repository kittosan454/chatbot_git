import json
import sqlite3


conn = sqlite3.connect('./data/FinalTwitchDb.db')
cur = conn.cursor()
cur.executemany("""
    INSERT INTO chat (content, writer_id, w_time, url_id)
    VALUES (?, ?, ?, '37089981')
""", chatdata)
conn.commit()
print(cur)