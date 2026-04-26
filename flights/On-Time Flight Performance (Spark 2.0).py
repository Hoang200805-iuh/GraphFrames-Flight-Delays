# Databricks notebook source
# MAGIC %md
# MAGIC *If you see ![](http://training.databricks.com/databricks_guide/ImportNotebookIcon3.png) at the top-left or top-right, click on the link to import this notebook in order to run it.*

# COMMAND ----------

# MAGIC %md
# MAGIC # On-Time Flight Performance with GraphFrames for Apache Spark
# MAGIC This notebook provides an analysis of On-Time Flight Performance and Departure Delays data using GraphFrames for Apache Spark.  This notebook has been updated to use Apache Spark 2.0 and GraphFrames 0.3.
# MAGIC * Original blog post: [On-Time Flight Performance with GraphFrames with Apache Spark Blog Post](https://databricks.com/blog/2016/03/16/on-time-flight-performance-with-graphframes-for-apache-spark.html)
# MAGIC * Original Notebook: [On-Time Flight Performance with GraphFrames with Apache Spark Notebook](http://cdn2.hubspot.net/hubfs/438089/notebooks/Samples/Miscellaneous/On-Time_Flight_Performance.html)
# MAGIC
# MAGIC
# MAGIC Source Data: 
# MAGIC * [OpenFlights: Airport, airline and route data](http://openflights.org/data.html)
# MAGIC * [United States Department of Transportation: Bureau of Transportation Statistics (TranStats)](http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time)
# MAGIC  * Note, the data used here was extracted from the US DOT:BTS between 1/1/2014 and 3/31/2014*
# MAGIC
# MAGIC References:
# MAGIC * [GraphFrames User Guide](http://graphframes.github.io/user-guide.html)
# MAGIC * [GraphFrames: DataFrame-based Graphs (GitHub)](https://github.com/graphframes/graphframes)
# MAGIC * [D3 Airports Example](http://mbostock.github.io/d3/talk/20111116/airports.html)
# MAGIC * [MLlib and Machine Learning: Binary Classification](https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparation
# MAGIC Extract the Airports and Departure Delays information from S3 / DBFS

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/flights/

# COMMAND ----------

# Set File Paths
tripdelaysFilePath = "/databricks-datasets/flights/departuredelays.csv"
airportsnaFilePath = "/databricks-datasets/flights/airport-codes-na.txt"

# Obtain airports dataset
airportsna = spark.read.csv(airportsnaFilePath, header='true', inferSchema='true', sep='\t')
airportsna.createOrReplaceTempView("airports_na")

# Obtain departure Delays data
departureDelays = spark.read.csv(tripdelaysFilePath, header='true')
departureDelays.createOrReplaceTempView("departureDelays")
departureDelays.cache()

# Available IATA codes from the departuredelays sample dataset
tripIATA = spark.sql("select distinct iata from (select distinct origin as iata from departureDelays union all select distinct destination as iata from departureDelays) a")
tripIATA.createOrReplaceTempView("tripIATA")

# Only include airports with atleast one trip from the departureDelays dataset
airports = spark.sql("select f.IATA, f.City, f.State, f.Country from airports_na f join tripIATA t on t.IATA = f.IATA")
airports.createOrReplaceTempView("airports")
airports.cache()

# COMMAND ----------

departureDelays.count()

# COMMAND ----------

# Build `departureDelays_geo` DataFrame
#  Obtain key attributes such as Date of flight, delays, distance, and airport information (Origin, Destination)  
departureDelays_geo = spark.sql("select cast(f.date as int) as tripid, cast(concat(concat(concat(concat(concat(concat('2014-', concat(concat(substr(cast(f.date as string), 1, 2), '-')), substr(cast(f.date as string), 3, 2)), ' '), substr(cast(f.date as string), 5, 2)), ':'), substr(cast(f.date as string), 7, 2)), ':00') as timestamp) as `localdate`, cast(f.delay as int), cast(f.distance as int), f.origin as src, f.destination as dst, o.city as city_src, d.city as city_dst, o.state as state_src, d.state as state_dst from departuredelays f join airports o on o.iata = f.origin join airports d on d.iata = f.destination") 

# Create Temporary View and cache
departureDelays_geo.createOrReplaceTempView("departureDelays_geo")
departureDelays_geo.cache()

# Count
departureDelays_geo.count()

# COMMAND ----------

# Review the top 10 rows of the `departureDelays_geo` DataFrame
departureDelays_geo.show(10)

# COMMAND ----------

# Using `display` to view the data
display(departureDelays_geo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building the Graph
# MAGIC Now that we've imported our data, we're going to need to build our graph. To do so we're going to do two things: we are going to build the structure of the vertices (or nodes) and we're going to build the structure of the edges. What's awesome about GraphFrames is that this process is incredibly simple. 
# MAGIC * Rename IATA airport code to **id** in the Vertices Table
# MAGIC * Start and End airports to **src** and **dst** for the Edges Table (flights)
# MAGIC
# MAGIC These are required naming conventions for vertices and edges in GraphFrames as of the time of this writing (Feb. 2016).

# COMMAND ----------

# MAGIC %md
# MAGIC **WARNING:** If the graphframes package, required in the cell below, is not installed, follow the instructions [here](http://cdn2.hubspot.net/hubfs/438089/notebooks/help/Setup_graphframes_package.html).

# COMMAND ----------

# Note, ensure you have already installed the GraphFrames spack-package
from pyspark.sql.functions import *
from graphframes import *

# Create Vertices (airports) and Edges (flights)
tripVertices = airports.withColumnRenamed("IATA", "id").distinct()
tripEdges = departureDelays_geo.select("tripid", "delay", "src", "dst", "city_dst", "state_dst")

# Cache Vertices and Edges
tripEdges.cache()
tripVertices.cache()

# COMMAND ----------

# Vertices
#   The vertices of our graph are the airports
display(tripVertices)

# COMMAND ----------

# Edges
#  The edges of our graph are the flights between airports
display(tripEdges)

# COMMAND ----------

# Build `tripGraph` GraphFrame
#  This GraphFrame builds up on the vertices and edges based on our trips (flights)
tripGraph = GraphFrame(tripVertices, tripEdges)

# Build `tripGraphPrime` GraphFrame
#   This graphframe contains a smaller subset of data to make it easier to display motifs and subgraphs (below)
tripEdgesPrime = departureDelays_geo.select("tripid", "delay", "src", "dst")
tripGraphPrime = GraphFrame(tripVertices, tripEdgesPrime)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Queries
# MAGIC Let's start with a set of simple graph queries to understand flight performance and departure delays

# COMMAND ----------

# MAGIC %md
# MAGIC #### Determine the number of airports and trips

# COMMAND ----------

print "Airports: %d" % tripGraph.vertices.count()
print "Trips: %d" % tripGraph.edges.count()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Determining the longest delay in this dataset

# COMMAND ----------

tripGraph.edges.groupBy().max("delay").show()

# COMMAND ----------

# Finding the longest Delay
longestDelay = tripGraph.edges.groupBy().max("delay")
display(longestDelay)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Determining the number of delayed vs. on-time / early flights

# COMMAND ----------

# Determining number of on-time / early flights vs. delayed flights
print "On-time / Early Flights: %d" % tripGraph.edges.filter("delay <= 0").count()
print "Delayed Flights: %d" % tripGraph.edges.filter("delay > 0").count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### What flights departing SEA are most likely to have significant delays
# MAGIC Note, delay can be <= 0 meaning the flight left on time or early

# COMMAND ----------

tripGraph.edges\
  .filter("src = 'SEA' and delay > 0")\
  .groupBy("src", "dst")\
  .avg("delay")\
  .sort(desc("avg(delay)"))\
  .show(5)
  

# COMMAND ----------

display(tripGraph.edges.filter("src = 'SEA' and delay > 0").groupBy("src", "dst").avg("delay").sort(desc("avg(delay)")))

# COMMAND ----------

# MAGIC %md
# MAGIC #### What destinations tend to have delays

# COMMAND ----------

# After displaying tripDelays, use Plot Options to set `state_dst` as a Key.
tripDelays = tripGraph.edges.filter("delay > 0")
display(tripDelays)

# COMMAND ----------

# MAGIC %md
# MAGIC #### What destinations tend to have significant delays departing from SEA

# COMMAND ----------

# States with the longest cumulative delays (with individual delays > 100 minutes) (origin: Seattle)
display(tripGraph.edges.filter("src = 'SEA' and delay > 100"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vertex Degrees
# MAGIC * `inDegrees`: Incoming connections to the airport
# MAGIC * `outDegrees`: Outgoing connections from the airport 
# MAGIC * `degrees`: Total connections to and from the airport
# MAGIC
# MAGIC Reviewing the various properties of the property graph to understand the incoming and outgoing connections between airports.

# COMMAND ----------

# Degrees
#  The number of degrees - the number of incoming and outgoing connections - for various airports within this sample dataset
display(tripGraph.degrees.sort(desc("degree")).limit(20))

# COMMAND ----------

# inDegrees
#  The number of degrees - the number of incoming connections - for various airports within this sample dataset
display(tripGraph.inDegrees.sort(desc("inDegree")).limit(20))

# COMMAND ----------

# outDegrees
#  The number of degrees - the number of outgoing connections - for various airports within this sample dataset
display(tripGraph.outDegrees.sort(desc("outDegree")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## City / Flight Relationships through Motif Finding
# MAGIC To more easily understand the complex relationship of city airports and their flights with each other, we can use motifs to find patterns of airports (i.e. vertices) connected by flights (i.e. edges). The result is a DataFrame in which the column names are given by the motif keys.

# COMMAND ----------

# MAGIC %md
# MAGIC #### What delays might we blame on SFO

# COMMAND ----------

# Using tripGraphPrime to more easily display 
#   - The associated edge (ab, bc) relationships 
#   - With the different the city / airports (a, b, c) where SFO is the connecting city (b)
#   - Ensuring that flight ab (i.e. the flight to SFO) occured before flight bc (i.e. flight leaving SFO)
#   - Note, TripID was generated based on time in the format of MMDDHHMM converted to int
#       - Therefore bc.tripid < ab.tripid + 10000 means the second flight (bc) occured within approx a day of the first flight (ab)
# Note: In reality, we would need to be more careful to link trips ab and bc.
motifs = tripGraphPrime.find("(a)-[ab]->(b); (b)-[bc]->(c)")\
  .filter("(b.id = 'SFO') and (ab.delay > 500 or bc.delay > 500) and bc.tripid > ab.tripid and bc.tripid < ab.tripid + 10000")
display(motifs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Determining Airport Ranking using PageRank
# MAGIC There are a large number of flights and connections through these various airports included in this Departure Delay Dataset.  Using the `pageRank` algorithm, Spark iteratively traverses the graph and determines a rough estimate of how important the airport is.

# COMMAND ----------

# Determining Airport ranking of importance using `pageRank`
ranks = tripGraph.pageRank(resetProbability=0.15, maxIter=5)
display(ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Most popular flights (single city hops)
# MAGIC Using the `tripGraph`, we can quickly determine what are the most popular single city hop flights

# COMMAND ----------

# Determine the most popular flights (single city hops)
import pyspark.sql.functions as func
topTrips = tripGraph \
  .edges \
  .groupBy("src", "dst") \
  .agg(func.count("delay").alias("trips")) 

# COMMAND ----------

# Show the top 20 most popular flights (single city hops)
display(topTrips.orderBy(topTrips.trips.desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top Transfer Cities
# MAGIC Many airports are used as transfer points instead of the final Destination.  An easy way to calculate this is by calculating the ratio of inDegree (the number of flights to the airport) / outDegree (the number of flights leaving the airport).  Values close to 1 may indicate many transfers, whereas values < 1 indicate many outgoing flights and > 1 indicate many incoming flights.  Note, this is a simple calculation that does not take into account of timing or scheduling of flights, just the overall aggregate number within the dataset.

# COMMAND ----------

# Calculate the inDeg (flights into the airport) and outDeg (flights leaving the airport)
inDeg = tripGraph.inDegrees
outDeg = tripGraph.outDegrees

# Calculate the degreeRatio (inDeg/outDeg)
degreeRatio = inDeg.join(outDeg, inDeg.id == outDeg.id) \
  .drop(outDeg.id) \
  .selectExpr("id", "double(inDegree)/double(outDegree) as degreeRatio") \
  .cache()

# Join back to the `airports` DataFrame (instead of registering temp table as above)
nonTransferAirports = degreeRatio.join(airports, degreeRatio.id == airports.IATA) \
  .selectExpr("id", "city", "degreeRatio") \
  .filter("degreeRatio < .9 or degreeRatio > 1.1")

# List out the city airports which have abnormal degree ratios.
display(nonTransferAirports)

# COMMAND ----------

# Join back to the `airports` DataFrame (instead of registering temp table as above)
transferAirports = degreeRatio.join(airports, degreeRatio.id == airports.IATA) \
  .selectExpr("id", "city", "degreeRatio") \
  .filter("degreeRatio between 0.9 and 1.1")
  
# List out the top 10 transfer city airports
display(transferAirports.orderBy("degreeRatio").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Breadth First Search 
# MAGIC Breadth-first search (BFS) is designed to traverse the graph to quickly find the desired vertices (i.e. airports) and edges (i.e flights).  Let's try to find the shortest number of connections between cities based on the dataset.  Note, these examples do not take into account of time or distance, just hops between cities.

# COMMAND ----------

# Example 1: Direct Seattle to San Francisco 
filteredPaths = tripGraph.bfs(
  fromExpr = "id = 'SEA'",
  toExpr = "id = 'SFO'",
  maxPathLength = 1)
display(filteredPaths)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, there are a number of direct flights between Seattle and San Francisco.

# COMMAND ----------

# Example 2: Direct San Francisco and Buffalo
filteredPaths = tripGraph.bfs(
  fromExpr = "id = 'SFO'",
  toExpr = "id = 'BUF'",
  maxPathLength = 2)
display(filteredPaths)

# COMMAND ----------

# Display most popular layover cities by descending count
display(filteredPaths.groupBy("v1.id", "v1.City").count().orderBy(desc("count")).limit(10))

# COMMAND ----------

# Example 2: Direct San Francisco and Buffalo
filteredPaths = tripGraph.bfs(
  fromExpr = "id = 'SFO'",
  toExpr = "id = 'BUF'",
  maxPathLength = 1)
display(filteredPaths)

# COMMAND ----------

# MAGIC %md
# MAGIC But there are no direct flights between San Francisco and Buffalo.

# COMMAND ----------

# Example 2a: Flying from San Francisco to Buffalo
filteredPaths = tripGraph.bfs(
  fromExpr = "id = 'SFO'",
  toExpr = "id = 'BUF'",
  maxPathLength = 2)
display(filteredPaths)

# COMMAND ----------

# MAGIC %md
# MAGIC But there are flights from San Francisco to Buffalo with Minneapolis as the transfer point.  But what are the most popular layovers between `SFO` and `BUF`?

# COMMAND ----------

# Display most popular layover cities by descending count
display(filteredPaths.groupBy("v1.id", "v1.City").count().orderBy(desc("count")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the D3 Visualization
# MAGIC Using the airports D3 visualization to visualize airports and flight paths

# COMMAND ----------

# MAGIC %scala
# MAGIC package d3a
# MAGIC // We use a package object so that we can define top level classes like Edge that need to be used in other cells
# MAGIC
# MAGIC import org.apache.spark.sql._
# MAGIC import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML
# MAGIC
# MAGIC case class Edge(src: String, dest: String, count: Long)
# MAGIC
# MAGIC case class Node(name: String)
# MAGIC case class Link(source: Int, target: Int, value: Long)
# MAGIC case class Graph(nodes: Seq[Node], links: Seq[Link])
# MAGIC
# MAGIC object graphs {
# MAGIC val sqlContext = SQLContext.getOrCreate(org.apache.spark.SparkContext.getOrCreate())
# MAGIC import sqlContext.implicits._
# MAGIC
# MAGIC def force(clicks: Dataset[Edge], height: Int = 100, width: Int = 960): Unit = {
# MAGIC   val data = clicks.collect()
# MAGIC   val nodes = (data.map(_.src) ++ data.map(_.dest)).map(_.replaceAll("_", " ")).toSet.toSeq.map(Node)
# MAGIC   val links = data.map { t =>
# MAGIC     Link(nodes.indexWhere(_.name == t.src.replaceAll("_", " ")), nodes.indexWhere(_.name == t.dest.replaceAll("_", " ")), t.count / 20 + 1)
# MAGIC   }
# MAGIC   showGraph(height, width, Seq(Graph(nodes, links)).toDF().toJSON.first())
# MAGIC }
# MAGIC
# MAGIC /**
# MAGIC  * Displays a force directed graph using d3
# MAGIC  * input: {"nodes": [{"name": "..."}], "links": [{"source": 1, "target": 2, "value": 0}]}
# MAGIC  */
# MAGIC def showGraph(height: Int, width: Int, graph: String): Unit = {
# MAGIC
# MAGIC displayHTML(s"""<!DOCTYPE html>
# MAGIC <html>
# MAGIC   <head>
# MAGIC     <link type="text/css" rel="stylesheet" href="https://mbostock.github.io/d3/talk/20111116/style.css"/>
# MAGIC     <style type="text/css">
# MAGIC       #states path {
# MAGIC         fill: #ccc;
# MAGIC         stroke: #fff;
# MAGIC       }
# MAGIC
# MAGIC       path.arc {
# MAGIC         pointer-events: none;
# MAGIC         fill: none;
# MAGIC         stroke: #000;
# MAGIC         display: none;
# MAGIC       }
# MAGIC
# MAGIC       path.cell {
# MAGIC         fill: none;
# MAGIC         pointer-events: all;
# MAGIC       }
# MAGIC
# MAGIC       circle {
# MAGIC         fill: steelblue;
# MAGIC         fill-opacity: .8;
# MAGIC         stroke: #fff;
# MAGIC       }
# MAGIC
# MAGIC       #cells.voronoi path.cell {
# MAGIC         stroke: brown;
# MAGIC       }
# MAGIC
# MAGIC       #cells g:hover path.arc {
# MAGIC         display: inherit;
# MAGIC       }
# MAGIC     </style>
# MAGIC   </head>
# MAGIC   <body>
# MAGIC     <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.js"></script>
# MAGIC     <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.csv.js"></script>
# MAGIC     <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.geo.js"></script>
# MAGIC     <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.geom.js"></script>
# MAGIC     <script>
# MAGIC       var graph = $graph;
# MAGIC       var w = $width;
# MAGIC       var h = $height;
# MAGIC
# MAGIC       var linksByOrigin = {};
# MAGIC       var countByAirport = {};
# MAGIC       var locationByAirport = {};
# MAGIC       var positions = [];
# MAGIC
# MAGIC       var projection = d3.geo.azimuthal()
# MAGIC           .mode("equidistant")
# MAGIC           .origin([-98, 38])
# MAGIC           .scale(1400)
# MAGIC           .translate([640, 360]);
# MAGIC
# MAGIC       var path = d3.geo.path()
# MAGIC           .projection(projection);
# MAGIC
# MAGIC       var svg = d3.select("body")
# MAGIC           .insert("svg:svg", "h2")
# MAGIC           .attr("width", w)
# MAGIC           .attr("height", h);
# MAGIC
# MAGIC       var states = svg.append("svg:g")
# MAGIC           .attr("id", "states");
# MAGIC
# MAGIC       var circles = svg.append("svg:g")
# MAGIC           .attr("id", "circles");
# MAGIC
# MAGIC       var cells = svg.append("svg:g")
# MAGIC           .attr("id", "cells");
# MAGIC
# MAGIC       var arc = d3.geo.greatArc()
# MAGIC           .source(function(d) { return locationByAirport[d.source]; })
# MAGIC           .target(function(d) { return locationByAirport[d.target]; });
# MAGIC
# MAGIC       d3.select("input[type=checkbox]").on("change", function() {
# MAGIC         cells.classed("voronoi", this.checked);
# MAGIC       });
# MAGIC
# MAGIC       // Draw US map.
# MAGIC       d3.json("https://mbostock.github.io/d3/talk/20111116/us-states.json", function(collection) {
# MAGIC         states.selectAll("path")
# MAGIC           .data(collection.features)
# MAGIC           .enter().append("svg:path")
# MAGIC           .attr("d", path);
# MAGIC       });
# MAGIC
# MAGIC       // Parse links
# MAGIC       graph.links.forEach(function(link) {
# MAGIC         var origin = graph.nodes[link.source].name;
# MAGIC         var destination = graph.nodes[link.target].name;
# MAGIC
# MAGIC         var links = linksByOrigin[origin] || (linksByOrigin[origin] = []);
# MAGIC         links.push({ source: origin, target: destination });
# MAGIC
# MAGIC         countByAirport[origin] = (countByAirport[origin] || 0) + 1;
# MAGIC         countByAirport[destination] = (countByAirport[destination] || 0) + 1;
# MAGIC       });
# MAGIC
# MAGIC       d3.csv("https://mbostock.github.io/d3/talk/20111116/airports.csv", function(data) {
# MAGIC
# MAGIC         // Build list of airports.
# MAGIC         var airports = graph.nodes.map(function(node) {
# MAGIC           return data.find(function(airport) {
# MAGIC             if (airport.iata === node.name) {
# MAGIC               var location = [+airport.longitude, +airport.latitude];
# MAGIC               locationByAirport[airport.iata] = location;
# MAGIC               positions.push(projection(location));
# MAGIC
# MAGIC               return true;
# MAGIC             } else {
# MAGIC               return false;
# MAGIC             }
# MAGIC           });
# MAGIC         });
# MAGIC
# MAGIC         // Compute the Voronoi diagram of airports' projected positions.
# MAGIC         var polygons = d3.geom.voronoi(positions);
# MAGIC
# MAGIC         var g = cells.selectAll("g")
# MAGIC             .data(airports)
# MAGIC           .enter().append("svg:g");
# MAGIC
# MAGIC         g.append("svg:path")
# MAGIC             .attr("class", "cell")
# MAGIC             .attr("d", function(d, i) { return "M" + polygons[i].join("L") + "Z"; })
# MAGIC             .on("mouseover", function(d, i) { d3.select("h2 span").text(d.name); });
# MAGIC
# MAGIC         g.selectAll("path.arc")
# MAGIC             .data(function(d) { return linksByOrigin[d.iata] || []; })
# MAGIC           .enter().append("svg:path")
# MAGIC             .attr("class", "arc")
# MAGIC             .attr("d", function(d) { return path(arc(d)); });
# MAGIC
# MAGIC         circles.selectAll("circle")
# MAGIC             .data(airports)
# MAGIC             .enter().append("svg:circle")
# MAGIC             .attr("cx", function(d, i) { return positions[i][0]; })
# MAGIC             .attr("cy", function(d, i) { return positions[i][1]; })
# MAGIC             .attr("r", function(d, i) { return Math.sqrt(countByAirport[d.iata]); })
# MAGIC             .sort(function(a, b) { return countByAirport[b.iata] - countByAirport[a.iata]; });
# MAGIC       });
# MAGIC     </script>
# MAGIC   </body>
# MAGIC </html>""")
# MAGIC   }
# MAGIC
# MAGIC   def help() = {
# MAGIC displayHTML("""
# MAGIC <p>
# MAGIC Produces a force-directed graph given a collection of edges of the following form:</br>
# MAGIC <tt><font color="#a71d5d">case class</font> <font color="#795da3">Edge</font>(<font color="#ed6a43">src</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">dest</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">count</font>: <font color="#a71d5d">Long</font>)</tt>
# MAGIC </p>
# MAGIC <p>Usage:<br/>
# MAGIC <tt>%scala</tt></br>
# MAGIC <tt><font color="#a71d5d">import</font> <font color="#ed6a43">d3._</font></tt><br/>
# MAGIC <tt><font color="#795da3">graphs.force</font>(</br>
# MAGIC &nbsp;&nbsp;<font color="#ed6a43">height</font> = <font color="#795da3">500</font>,<br/>
# MAGIC &nbsp;&nbsp;<font color="#ed6a43">width</font> = <font color="#795da3">500</font>,<br/>
# MAGIC &nbsp;&nbsp;<font color="#ed6a43">clicks</font>: <font color="#795da3">Dataset</font>[<font color="#795da3">Edge</font>])</tt>
# MAGIC </p>""")
# MAGIC   }
# MAGIC }

# COMMAND ----------

# MAGIC %scala d3a.graphs.help()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize On-time and Early Arrivals

# COMMAND ----------

# MAGIC %scala
# MAGIC // On-time and Early Arrivals
# MAGIC import d3a._
# MAGIC graphs.force(
# MAGIC   height = 800,
# MAGIC   width = 1200,
# MAGIC   clicks = sql("""select src, dst as dest, count(1) as count from departureDelays_geo where delay <= 0 group by src, dst""").as[Edge])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize Delayed Trips Departing from the West Coast
# MAGIC
# MAGIC Notice that most of the delayed trips are with Western US cities

# COMMAND ----------

# MAGIC %scala
# MAGIC // Delayed Trips from CA, OR, and/or WA
# MAGIC import d3a._
# MAGIC graphs.force(
# MAGIC   height = 800,
# MAGIC   width = 1200,
# MAGIC   clicks = sql("""select src, dst as dest, count(1) as count from departureDelays_geo where state_src in ('CA', 'OR', 'WA') and delay > 0 group by src, dst""").as[Edge])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize All Flights (from this dataset)

# COMMAND ----------

# MAGIC %scala
# MAGIC // Trips (from DepartureDelays Dataset)
# MAGIC import d3a._
# MAGIC graphs.force(
# MAGIC   height = 800,
# MAGIC   width = 1200,
# MAGIC   clicks = sql("""select src, dst as dest, count(1) as count from departureDelays_geo group by src, dst""").as[Edge])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting Flight Delays
# MAGIC Extending upon analysis we have done up to this point, can we also predict if a flight will be delayed, on-time, or early based on the available data.
# MAGIC
# MAGIC ### Prepare the Dataset
# MAGIC The first thing we will do is to cleanse the data and apply some labels to our information (e.g. early, on-time, delayed).  As well, we will want to remove any rows with NULL values.

# COMMAND ----------

# This contains a generated mapping between tripid and airline
#   You can get the file at https://github.com/dennyglee/databricks/blob/master/misc/trip_airline_map.csv
#   For this example, the trip_airline_map.csv file has been pushed to in my mounted bucket.
tripAirlineMap = spark.read.csv("/mnt/tardis6/departuredelays/trip_airline_map.csv", sep=",", header=True)
tripAirlineMap.createOrReplaceTempView("tripAirlineMap")

# COMMAND ----------

# Prep dataset
# Limiting to only Las Vegas (LAS) and Seattle (SEA) predictions so it can run on Databricks Community Edition
flightML = spark.sql("select cast(distance as double) as distance, src as origin, state_src as origin_state, dst as destination, state_dst as destination_state, concat(concat(concat(cast(tripid as string), src), dst), cast((delay + 2000) as string)) as trip_identifier, case when delay <= 0 then 'on-time' else 'delayed' end as flight_status from departureDelays_geo where src in ('LAS', 'SEA')")
flightML = flightML.dropna().dropDuplicates()
flightML.createOrReplaceTempView("flightML")

# COMMAND ----------

# Join flights and airline information
dataset = spark.sql("select f.distance, f.origin, f.origin_state, f.destination, f.destination_state, f.trip_identifier, f.flight_status, m.airline from flightML f join tripAirlineMap m on m.trip_identifier = f.trip_identifier")
dataset = dataset.dropDuplicates()
#dataset = flightML
cols = dataset.columns

# COMMAND ----------

dataset.printSchema()

# COMMAND ----------

dataset.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Building ML Pipeline
# MAGIC Before we can run our various models against this data, we will first need to vectorize our data via One-Hot Encorder (for category data), String Indexer (create an index based on our labelled values), and Vector Assembler.

# COMMAND ----------

# One-Hot Encoding
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

categoricalColumns = ["origin", "origin_state", "destination", "destination_state", "trip_identifier", "airline"]
#categoricalColumns = ["origin", "origin_state", "destination", "destination_state", "trip_identifier"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
  # Category Indexing with StringIndexer
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
  
  # Use OneHotEncoder to convert categorical variables into binary SparseVectors
  encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
  
  # Add stages.  These are not run here, but will run all at once later on.
  stages += [stringIndexer, encoder]

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol = "flight_status", outputCol = "label")
stages += [label_stringIdx]

# Transform all features into a vector using VectorAssembler
numericCols = ["distance"]
assemblerInputs = map(lambda c: c + "classVec", categoricalColumns) + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

# Create a Pipeline.
pipeline = Pipeline(stages=stages)
# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
pipelineModel = pipeline.fit(dataset)
dataset = pipelineModel.transform(dataset)

# Keep relevant columns
selectedcols = ["label", "features"] + cols
dataset = dataset.select(selectedcols)
display(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Randomly split data into training and test datasets
# MAGIC * Set the seed for reproducibility

# COMMAND ----------

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print trainingData.count()
print testData.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression
# MAGIC Let's try using logistic regression to see if we can accurately predict if a flight will be delayed.
# MAGIC * First, we will train the data using Logistic Regression
# MAGIC * Next we will run that model against the testData

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)

# COMMAND ----------

# MAGIC %md
# MAGIC ### View LR Model's predictions
# MAGIC * Recall, label is the actual test value, prediction is the predicted value
# MAGIC  * where 0 - on-time, 1 - delayed

# COMMAND ----------

selected = predictions.select("label", "prediction", "probability", "flight_status", "destination", "destination_state").where("destination = 'SEA'")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate our model
# MAGIC Let's use the `BinaryClassificationEvaluator` to determine the precision of our model.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

# COMMAND ----------

