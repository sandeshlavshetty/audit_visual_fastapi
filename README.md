# Team AI_Architects

## Model_2

🥈Query Data Visualizer
  [ Data Visualizer tool ](visual.png)
  ![visual](https://github.com/sandeshlavshetty/AI_Architects/assets/138968398/f2c59efa-3637-45d2-82b7-5e24cdf4fb96)

  Input :- 
   1. Any type of Banking and non-banking data which Banker/auditers/Heads want to see in visual presentation .
   2. Types of Diagram ( histogram,piecharts,etc.)
   3. Querry for which data visual user want through chat interface only in natural language.

  Output :- 
   1. It intially create self querry and create visual presentations of Data.
   2. The query can be given by user  and create on demand visual presentation with hassle free work.  ( e.g. create a line plot between loan approval and loan re-payments , create histogram for types of loans approved,etc)

    Working :-
     1. Data is first splitted into chunks and there respectivev context is stored in DB using our B-LLM.
     2. Based on User query a RAG(Retrieval-Augmented Generation) is performed and data is forwarded to viz-Generator.
     3. Viz-Generator ( Visual Generator ) generate the images and graphs based on Related data and showed at info-grapher stage.
     4. At info-grapher stage the Data-visualis are presented and further customizations are done ( e.g.* Sketch prompt: line sketch art,line drawing,etc ) through chat interface only in natural language.


     #### Data Safety 

Each feature of our solution is enabled with security layer of Qpaque layer 

 working :-

  1. ![Screenshot 2024-06-28 214835](https://github.com/sandeshlavshetty/AI_Architects/assets/138968398/b5583260-e7e1-4d19-9303-a39aa267beb8)
  2. ![Screenshot 2024-06-28 214905](https://github.com/sandeshlavshetty/AI_Architects/assets/138968398/e34a228b-82b0-45da-a5df-afa954be892d)

This is achived through using tools like langchain and Microsoft Presidio


Thank-you.....
