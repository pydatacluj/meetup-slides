Required tools 
==============

Start with:
```
sudo sysctl -w vm.max_map_count=262144
docker-compose up -d
```
Connect to http://localhost:8080/  
`u:sql`, `p:changeme`, `db:sql`


And connect to http://localhost:5601/

Introduction
============
```
DELETE me
POST me/_doc
{
  "name": "Alpar Torok",
  "jobs" : [
    { "at" : "Elastic", "role" : "Software Enginer", "since" : "2018" },
    { "at" : "Freelancer", "role" : "Consulting Architect", "since" : "2016" },
    { "at" : "gTeam/crossover", "role" : "Chief Software Architect", "since" : "2014" },
    { "at" : "HP", "role" : "Functional Architect", "since" : "2014" },
    { "at" : "HP", "role" : "Software Enginer", "since" : "2010" },
    { "at" : "Bitdefender", "role" : "Automation Enginer", "since" : "2008" },
    { "at" : "Freelancer", "role" : "Web developer", "since" : "2006" }
  ],
  "passions" : [ "family", "job", "photography" ],
  "email" : "torokalpar@gmail.com",
  "linkedin": "https://www.linkedin.com/in/torokalpar/",
  "github": "https://github.com/atorok/elasticsearch-pydata-cluj"
}

GET me/_search
```

Demo
====

Insert  some data: 
```
DROP TABLE IF EXISTS users;
CREATE TABLE users
(
   name VARCHAR(100),
   comments VARCHAR(1000)
);
INSERT INTO users VALUES ('Lenard Palko', 'Went to college wiht Alpar');
INSERT INTO users VALUES ('Alpar Torok', 'Developer at elastic');
INSERT INTO users VALUES ('Alpár Gábos', 'Engineer at Facebook');
INSERT INTO users VALUES ('Alpar Alpar', 'Who is that guy?');
```

Check that we have it:
```
SELECT * FROM users;
```

User searches by name:
```
SELECT * FROM users WHERE name='Alpar';
```
(No results)


Can we do better?:
```
SELECT * FROM users WHERE name LIKE '%Alpar%';
```
(Alpar Torok, Alpar Alpar)


User searches by full name:
```
SELECT * FROM users WHERE name LIKE '%Torok Alpar%';
SELECT * FROM users WHERE name LIKE '%Torok%Alpar%';
SELECT * FROM users WHERE name LIKE '%Alpar Torok%';
```


How do we fix that?:
```
SELECT * FROM users
WHERE name LIKE '%alpar%' AND
      name LIKE '%torok%';
```
(No result, case sensitive)

```
SELECT * FROM users
WHERE name LIKE '%Alpar%' AND
      name LIKE '%Torok%';
```

Search with a typo: 
```
SELECT * FROM users WHERE name LIKE '%Albar%';

SELECT * FROM users 
WHERE name LIKE '%_lpar%' OR
      name LIKE '%A_par%' OR
      name LIKE '%Al_ar%' OR
      name LIKE '%Alp_r%' OR
      name LIKE '%Alpa_%' 
;
```

Search the entire db:
```
SELECT * FROM users 
WHERE name LIKE '%Alpar%' OR
      comments LIKE '%Alpar%';
```
Missing results, incorrect order


With Elasticsearch
==================


Start inserting some data:
```
wget http://download.elastic.co/workshops/basic-kibana/injector/injector-7.0.jar
time java -jar injector-7.0.jar -nb 10000000
```
Check on this lather;


```

### REINIT
DELETE user
PUT user
{
  "settings" : {
    "analysis" : {
      "analyzer" : {
        "my-asciifolding" : {
          "tokenizer" : "standard",
          "filter" : [
               "lowercase",
               "asciifolding"
            ]
        }
      }
    }
  },
  "mappings": {
      "properties": {
        "name": {
          "type": "text",
          "analyzer": "my-asciifolding"
        },
        "comments": {
          "type": "text",
          "analyzer": "my-asciifolding"
        }
      }
  }
}


POST user/_doc
{
  "name": "Lenard Palko",
  "comments": "Went to college wiht Alpar"
}
POST user/_doc
{
  "name": "Alpar Torok",
  "comments": "Developer at elastic"
}
POST user/_doc
{
  "name": "Alpár Gábos",
  "comments": "Engineer at Facebook"
}
POST user/_doc
{
  "name": "Alpar Alpar",
  "comments": "Who is that guy?"
}


GET user/_search
{
  "query": {
    "match": {
      "name": "Alpar"
    }
  }
}

GET user/_search
{
  "query": {
    "match": {
      "name": "AlPáR"
    }
  }
}

GET user/_search
{
  "query": {
    "multi_match" : {
      "query":    "Alpar",
      "fields": [ "name", "comments" ]
    }
  }
}

GET user/_search
{
  "query": {
    "multi_match" : {
      "query":    "Alpar",
      "fields": [ "name^3", "comments" ]
    }
  }
}

GET user/_search
{
  "query": {
    "multi_match" : {
      "query":    "Albar",
      "fields": [ "name^3", "comments" ],
      "fuzziness": 1
    }
  }
}
```


Look at an aggregation:

```
GET person/_search?track_total_hits=true
{
  "query": {
    "match": {
      "name": "joe"
    }
  },
  "size": 0,
  "aggs": {
    "by_year": {
      "date_histogram": {
        "field": "dateofbirth",
        "calendar_interval": "year"
      },
      "aggs": {
        "by_gender": {
          "terms": {
            "field": "gender"
          },
          "aggs": {
            "children": {
              "avg": {
                "field": "children"
              }
            }
          }
        }
      }
    }
  }
}
```

Import export-bbl-7.2.ndjson to Kibana.

Explore the dashboard.

Remove a node and observe the shards.
