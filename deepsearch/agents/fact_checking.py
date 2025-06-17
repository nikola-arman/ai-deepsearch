import json
from typing import Generator, List, Dict, Any, Set
import logging
from deepsearch.agents.faiss_indexing import init_embedding_model
from deepsearch.schemas.agents import SearchState, SearchResult
from deepsearch.utils.streaming import wrap_thought, to_chunk_data
from deepsearch.utils.misc import get_url_domain
from deepsearch.agents.deep_reasoning import init_reasoning_llm
from langchain.prompts import PromptTemplate
from collections import defaultdict
from json_repair import repair_json

# Set up logging
logger = logging.getLogger("deepsearch.factcheck")

def group_similar_statements(statements: List[str]) -> Dict[str, List[str]]:
    """
    Group similar statements together based on their semantic meaning using embeddings.
    
    Args:
        statements: List of statements to group
        
    Returns:
        Dictionary mapping representative statements to lists of similar statements
    """
    try:
        embedding_model = init_embedding_model()
        
        # Initialize embedding model
        from sklearn.cluster import DBSCAN
        import numpy as np

        # Generate embeddings for all statements
        embeddings = embedding_model.embed_documents(statements)
        
        # Cluster similar statements using DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=1).fit(embeddings)
        labels = clustering.labels_

        # Group statements by cluster
        clusters = defaultdict(list)
        for statement, label in zip(statements, labels):
            clusters[label].append(statement)
            
        # For each cluster, find the most central statement as representative
        grouped_statements = {}
        for label, cluster_statements in clusters.items():
            if len(cluster_statements) == 1:
                # If only one statement, use it as representative
                grouped_statements[cluster_statements[0]] = cluster_statements
            else:
                # Find most central statement in cluster
                cluster_embeddings = embedding_model.embed_documents(cluster_statements)
                centroid = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                representative_idx = np.argmin(distances)
                representative = cluster_statements[representative_idx]
                grouped_statements[representative] = cluster_statements

        return grouped_statements

    except Exception as e:
        logger.error(f"Error grouping statements: {str(e)}", exc_info=True)
        # Return each statement as its own group if grouping fails
        return {s: [s] for s in statements}

def verify_statement_group(representative: str, statements: List[str], sources: List[SearchResult]) -> Dict[str, Any]:
    """
    Verify a group of similar statements against their sources.
    
    Args:
        representative: The representative statement for the group
        statements: All statements in the group
        sources: The search results that contain these statements
        
    Returns:
        Dictionary containing verification results
    """
    try:
        # Initialize LLM for verification
        llm = init_reasoning_llm()

        # Create prompt for verification
        verification_prompt = """Verify the following statement against multiple sources.

STATEMENT: {representative}

SOURCES:
{sources}

For each source, analyze:
1. Whether the source supports the statement
2. The level of confidence in the support (high/medium/low)
3. Any contradictions or conflicting information
4. The source's credibility

Format the response as a JSON object with these fields:
{{
    "verification_status": "verified" or "contradicted" or "unverified",
    "confidence": "high" or "medium" or "low",
    "supporting_sources": ["url1", "url2", ...],
    "contradicting_sources": ["url3", "url4", ...],
    "notes": "Additional observations about the verification"
}}
"""

        # Format sources for the prompt
        formatted_sources = []
        for source in sources:
            formatted_sources.append(f"URL: {source.url}")
            formatted_sources.append(f"Content: {source.content}")
            formatted_sources.append("---")

        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["representative", "sources"],
            template=verification_prompt
        )

        print("verification_prompt:", prompt_template.invoke({
            "representative": representative,
            "sources": "\n".join(formatted_sources)    
        }))

        # Create the chain
        chain = prompt_template | llm

        # Get the response
        response = chain.invoke({
            "representative": representative,
            "sources": "\n".join(formatted_sources)
        })
        
        # Extract the content if it's a message object
        response_text = response.content if hasattr(response, 'content') else response
        
        # Parse the response as JSON
        verification = repair_json(response_text, return_objects=True)  # Using eval since we trust the LLM output
        
        return verification

    except Exception as e:
        logger.error(f"Error verifying statement: {str(e)}", exc_info=True)
        return {
            "verification_status": "error",
            "confidence": "low",
            "supporting_sources": [],
            "contradicting_sources": [],
            "notes": f"Error during verification: {str(e)}"
        }


def verify_statements_of_source(source: SearchResult, other_sources: List[SearchResult]) -> Dict[str, Any]:
    """
    Verify statements of a source against other sources.
    
    Args:
        source: The source whose statements need to be verified
        other_sources: List of other sources to verify against
        
    Returns:
        Dictionary containing verification results with the following structure:
        {
            "source_url": str,
            "verification_status": "verified" | "partially_verified" | "contradicted" | "unverified",
            "confidence": "high" | "medium" | "low",
            "statements": [
                {
                    "statement": str,
                    "status": "verified" | "contradicted" | "unverified",
                    "supporting_sources": List[ { "statement": str, "source": str }],
                    "contradicting_sources": List[ { "statement": str, "source": str }],
                    "confidence": "high" | "medium" | "low",
                    "notes": str
                }
            ],
            "overall_notes": str
        }
    """
    try:
        if not source.extracted_information:
            return {
                "source_url": source.url,
                "verification_status": "unverified",
                "confidence": "low",
                "statements": [],
                "overall_notes": "No statements found in source"
            }

        # Initialize LLM for verification
        llm = init_reasoning_llm()

        # Create prompt for statement verification
        verification_prompt = """Verify the following statements from a source against other sources.

SOURCE STATEMENTS:
{statements}

OTHER SOURCES:
{other_sources}

For each statement, analyze:
1. Whether other sources support or contradict the statement
2. The level of confidence in the verification (high/medium/low)
3. Any contradictions or conflicting information
4. The credibility of supporting/contradicting sources

Format the response as a JSON object with these fields:
{{
    "statements": [
        {{
            "statement": "original statement",
            "status": "verified" or "contradicted" or "unverified",
            "supporting_sources": [{{ "statement": "statement1", "source": "url1" }}, {{ "statement": "statement2", "source": "url2" }}, ...],
            "contradicting_sources": [{{ "statement": "statement1", "source": "url1" }}, {{ "statement": "statement2", "source": "url2" }}, ...],
            "confidence": "high" or "medium" or "low",
            "notes": "Additional observations about the verification"
        }}
    ],
    "overall_status": "verified" or "partially_verified" or "contradicted" or "unverified",
    "overall_confidence": "high" or "medium" or "low",
    "overall_notes": "Summary of verification findings"
}}
"""

        # Format statements and sources for the prompt
        formatted_statements = "\n".join(f"- {s}" for s in source.extracted_information)
        
        formatted_sources = []
        for other_source in other_sources:
            formatted_sources.append(f"URL: {other_source.url}")
            if other_source.extracted_information:
                formatted_sources.append("Statements:")
                formatted_sources.extend(f"- {s}" for s in other_source.extracted_information)
            formatted_sources.append("---")

        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["statements", "other_sources"],
            template=verification_prompt
        )

        # Get the response
        response = llm.invoke(prompt_template.format(
            statements=formatted_statements,
            other_sources="\n".join(formatted_sources)
        ))
        
        # Extract the content if it's a message object
        response_text = response.content if hasattr(response, 'content') else response
        
        # Parse the response as JSON
        verification = repair_json(response_text, return_objects=True)

        # Add source URL to the result
        verification["source_url"] = source.url

        return verification

    except Exception as e:
        logger.error(f"Error verifying statements of source: {str(e)}", exc_info=True)
        return {
            "source_url": source.url,
            "verification_status": "error",
            "confidence": "low",
            "statements": [],
            "overall_notes": f"Error during verification: {str(e)}"
        }


def fact_checking_agent_legacy(state: SearchState) -> Generator[bytes, None, SearchState]:
    """
    Cross-validates extracted information across different sources.

    Args:
        state: The current search state with extracted information

    Returns:
        Updated state with verification results
    """
    # Check if we have results to analyze
    if not state.combined_results:
        return state

    try:
        # Collect all extracted information
        all_statements = []
        for result in state.combined_results:
            if result.extracted_information:
                all_statements.extend(result.extracted_information)

        if not all_statements:
            return state

        # Group similar statements
        statement_groups = group_similar_statements(all_statements)
        logger.info(f"Grouped statements into {len(statement_groups)} groups")

        # Initialize verification results
        state.verified_information = {
            "verified": [],
            "contradicted": [],
            "unverified": []
        }

        print("statement_groups:", statement_groups)

        # Verify each group
        for i, (representative, statements) in enumerate(statement_groups.items()):
            yield to_chunk_data(
                wrap_thought(
                    "Fact checking agent: Verifying group",
                    f"Verifying group {i+1}/{len(statement_groups)}: {representative[:100]}..."
                )
            )

            # Find sources that contain these statements
            relevant_sources = []
            for result in state.combined_results:
                if result.extracted_information and any(s in result.extracted_information for s in statements):
                    relevant_sources.append(result)

            # Verify the statement group
            verification = verify_statement_group(representative, statements, relevant_sources)

            print("verification:", verification)

            # Store verification results
            if verification["verification_status"] == "verified":
                state.verified_information["verified"].append({
                    "statement": representative,
                    "confidence": verification["confidence"],
                    "sources": verification["supporting_sources"],
                    "notes": verification["notes"]
                })
            elif verification["verification_status"] == "contradicted":
                state.verified_information["contradicted"].append({
                    "statement": representative,
                    "confidence": verification["confidence"],
                    "supporting_sources": verification["supporting_sources"],
                    "contradicting_sources": verification["contradicting_sources"],
                    "notes": verification["notes"]
                })
            else:
                state.verified_information["unverified"].append({
                    "statement": representative,
                    "confidence": verification["confidence"],
                    "sources": verification["supporting_sources"],
                    "notes": verification["notes"]
                })

        yield to_chunk_data(
            wrap_thought(
                "Fact checking agent: Complete",
                f"Verified {len(state.verified_information['verified'])} statements, "
                f"found {len(state.verified_information['contradicted'])} contradictions, "
                f"and {len(state.verified_information['unverified'])} unverified statements"
            )
        )

    except Exception as e:
        logger.error(f"Error in fact checking agent: {str(e)}", exc_info=True)
        yield to_chunk_data(
            wrap_thought(
                "Fact checking agent: Error",
                f"Error occurred during fact checking: {str(e)}"
            )
        )
        state.verified_information = {
            "verified": [],
            "contradicted": [],
            "unverified": []
        }

    return state


def sources_url_validation(sources: List[SearchResult]) -> List[bool]:
    """
    Validate a source based on its content and metadata.
    
    Args:
        sources: The list of search results containing the source content and metadata

    Returns:
        List of booleans indicating whether each source is valid
    """
    try:
        # Check if the source is a valid URL
        url_prefixes = []
        for source in sources:
            url_prefixes.append(get_url_domain(source.url))

        url_prefixes = list(set(url_prefixes))
        
        # Initialize LLM for validation
        llm = init_reasoning_llm()
        
        # Create prompt for URL validation with explicit format requirements
        validation_prompt = """Evaluate the credibility of these domains as information sources.
For each domain, respond with a single line in this exact format:
domain: True/False

Domains to evaluate:
{domains}

Guidelines:
- True: For well-known, reputable websites likely to provide accurate information
- False: For unknown, suspicious, or potentially unreliable sources
- Consider factors like domain age, reputation, and content quality
- If unsure, default to False for safety

Example response format:
example.com: True
unknown-site.com: False
"""

        # Get validation results from LLM
        results = llm.predict(validation_prompt.format(domains="\n".join(url_prefixes)))

        print("source_url_validation results:", results)
        
        # Parse results into boolean list with robust error handling
        validation_map = {}
        for line in results.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Try different parsing patterns
            if ":" in line:
                try:
                    domain, result = line.split(":", 1)
                    domain = domain.strip().lower()
                    result = result.strip().lower()
                    
                    # Handle various true/false representations
                    is_valid = result in ["true", "yes", "1", "valid", "reliable"]
                    validation_map[domain] = is_valid
                except Exception as e:
                    logger.warning(f"Failed to parse line '{line}': {str(e)}")
                    continue
        
        # If parsing failed for any domain, default to False
        return [validation_map.get(get_url_domain(source.url).lower(), False) for source in sources]
        
    except Exception as e:
        logger.error(f"Error in sources url validation: {str(e)}", exc_info=True)
        # Return all False in case of any error
        return [False] * len(sources)


def fact_checking_agent(state: SearchState) -> Generator[bytes, None, SearchState]:
    """
    Cross-validates extracted information across different sources.

    Args:
        state: The current search state with extracted information

    Returns:
        Updated state with verification results
    """
    # Check if we have results to analyze
    if not state.combined_results:
        return state

    try:
        # Collect all extracted information
        # all_statements = []
        # for result in state.combined_results:
        #     if result.extracted_information:
        #         all_statements.extend(result.extracted_information)

        # if not all_statements:
        #     return state

        # # Group similar statements
        # statement_groups = group_similar_statements(all_statements)
        # logger.info(f"Grouped statements into {len(statement_groups)} groups")

        # for source in state.combined_results:
        #     reduced_extracted_information = []
        #     for statement in source.extracted_information:
        #         if statement in statement_groups.keys():
        #             reduced_extracted_information.append(statement)
        #     source.extracted_information = reduced_extracted_information

        is_urls_credible = sources_url_validation(state.combined_results)
        for source, is_credible in zip(state.combined_results, is_urls_credible):
            source.is_url_credible = is_credible

        # Initialize verification results
        state.verified_information = {
            "verified": [],
            "contradicted": [],
            "unverified": []
        }

        for i, source in enumerate(state.combined_results):
            print("source_url:", source.url)
            if source.is_url_credible:
                for statement in source.extracted_information:
                    state.verified_information["verified"].append({
                        "statement": statement,
                        "title": source.title,
                        "source": source.url,
                        "confidence": "high",
                        "supporting_sources": [],
                        "notes": f"Credible source domain: {get_url_domain(source.url)}"
                    })
            else:
                try:
                    other_sources = [s for s in state.combined_results if get_url_domain(s.url) != get_url_domain(source.url)]
                    verification = verify_statements_of_source(source, other_sources)
                    print("verification:", json.dumps(verification, indent=2))

                    for statement in verification["statements"]:
                        # Store verification results
                        if statement.get("status", "") == "verified":
                            state.verified_information["verified"].append({
                                "statement": statement.get("statement", ""),
                                "title": source.title,
                                "source": source.url,
                                "confidence": statement.get("confidence", ""),
                                "supporting_sources": statement.get("supporting_sources", []),
                                "notes": statement.get("notes", "")
                            })
                        elif statement.get("status", "") == "contradicted":
                            state.verified_information["contradicted"].append({
                                "statement": statement.get("statement", ""),
                                "title": source.title,
                                "source": source.url,
                                "confidence": statement.get("confidence", ""),
                                "supporting_sources": statement.get("supporting_sources", []),
                                "contradicting_sources": statement.get("contradicting_sources", []),
                                "notes": statement.get("notes", "")
                            })
                        else:
                            state.verified_information["unverified"].append({
                                "statement": statement.get("statement", ""),
                                "title": source.title,
                                "source": source.url,
                                "confidence": statement.get("confidence", ""),
                                "supporting_sources": statement.get("supporting_sources", []),
                                "notes": statement.get("notes", "")
                            })
                except Exception as e:
                    logger.error(f"Error when fact checking source: {source.url}: {str(e)}", exc_info=True)
                    continue
        yield to_chunk_data(
            wrap_thought(
                "Fact checking agent: Complete",
                f"Verified {len(state.verified_information['verified'])} statements, "
                f"found {len(state.verified_information['contradicted'])} contradictions, "
                f"and {len(state.verified_information['unverified'])} unverified statements"
            )
        )

    except Exception as e:
        logger.error(f"Error in fact checking agent: {str(e)}", exc_info=True)
        yield to_chunk_data(
            wrap_thought(
                "Fact checking agent: Error",
                f"Error occurred during fact checking: {str(e)}"
            )
        )
        state.verified_information = {
            "verified": [],
            "contradicted": [],
            "unverified": []
        }

    return state


if __name__ == "__main__":
    test_statements = [
        'Ado is a Japanese singer who emerged as a significant figure in the music industry with her breakout single "Usseewa" in 2020.',
        'The single "Usseewa" is characterized by its raw, powerful vocals and bold lyrics, resonating with many young people in Japan by addressing societal frustrations.',
        'Ado further cemented her popularity by providing the singing voice for the character Uta in the 2022 anime film "One Piece Film: Red."',
        'The song "New Genesis" from the movie "One Piece Film: Red," sung by Ado, became a global hit.',
        'Ado is known for maintaining her anonymity by performing using an avatar and creative visual presentations, keeping her true identity hidden.',
        'From a young age, Ado was fascinated by Vocaloid music and the unique sound of synthesized voices, which led her to become an utaite, a singer who covers songs by other artists, particularly Vocaloid tracks.',
        'In 2024, Ado conducted a series of live activities including a SOLD OUT first world tour called Ado THE FIRST WORLD TOUR "Wish".',
        'Ado performed at Ado SPECIAL LIVE 2024 "Shinzou" held at Japan National Stadium, becoming the first female artist ever to perform at that venue.',
        'Ado also held the Ado JAPAN LIVE TOUR 2024 titled "Profile of Mona Lisa" in 2024.',
        'Ado released her latest studio album Zanmu in July 2024.',
        'Ado is scheduled to perform at the opening ceremony of Expo 2025 Osaka Kansai in April.',
        'Ado continues to attract further attention in 2024 with many exciting new releases lined up.',
        'Ado will headline the WORLD TOUR 2025 \'Hibana\'.',
        'Ado was born on October 24, 2002, in Tokyo, Japan.',
        'Ado is a Japanese singer and Utaite who is considered one of the most promising and popular young singers in Japan.',
        'Her third album, titled "Ado\'s Utattemita Album," was released in December 2023 and is a cover album featuring various Vocaloid and J-pop songs that fans voted on for her to cover.',
        'On April 20, 2024, Ado announced her second studio album, "Zanmu."',
        'The album "Zanmu" was released on July 10, 2024, alongside a live CD from her first world tour "Wish," which was recorded at Peacock Theater, California.',
        'In 2023, Ado continued releasing singles, her third album, and announced her first world tour.',
        'Notable singles released by Ado in 2023 include "Show" and "Kura Kura."',
        '"Show" quickly became one of Ado\'s most popular songs.',
        '"Kura Kura" was used as the opening song for the anime "Spy x Family," marking her first TV anime opening.',
        'Ado is a female utaite who began her activities on January 10, 2017 at the age of 14.',
        'In late 2020, Ado exploded onto the music scene with her debut release of "Usseewa" at just 17 years old.',
        'As of May 2024, "Usseewa" has over 340 million views on YouTube.',
        'As of May 2024, "Usseewa" has over 187 million plays on Spotify.',
        'Ado has been garnering lots of attention when releasing new covers or original music and also releases albums of her covered songs.',
        'Ado\'s vocal range displayed in "Aishite Aishite Aishite" is from C3 to G♯6.',
        'Ado\'s latest measured vocal range extends from a low and rumbling fried G♯2 all the way up to an ear-piercing screamed C7.',
        'Ado, born October 24, 2002, in Tokyo, Japan, is a Japanese singer who debuted in 2020 at age 17 with the digital single \'Usseewa.\'',
        'The digital single \'Usseewa\' topped several major Japanese music charts and achieved over 100 million YouTube views in under 150 days.',
        'In summer 2022, Ado contributed music to the \'One Piece\' film and released a highly successful album that broke numerous records.',
        'Ado\'s album saw seven of her songs simultaneously occupy the top seven spots on Apple Music\'s Japanese chart, a feat unprecedented for a Japanese artist.',
        'Information beyond 2022 is not available in this biography.',
        'Ado emerged as a significant figure in the music industry with her breakout single \'Usseewa\' in 2020.',
        'The single \'Usseewa\' is characterized by raw, powerful vocals and bold lyrics that resonated with many young people in Japan, addressing societal frustrations.',
        'Ado further cemented her popularity by providing the singing voice for Uta in the 2022 anime film \'One Piece Film: Red.\'',
        'The song \'New Genesis,\' performed by Ado for \'One Piece Film: Red,\' became a global hit.',
        'Ado is known for maintaining her anonymity by performing using an avatar and creative visual presentations, keeping her true identity hidden.',
        'From a young age, Ado was fascinated by Vocaloid music and the unique sound of synthesized voices, which led her to become an utaite, a singer who covers songs by other artists, particularly Vocaloid tracks.',
        'Ado was born on October 24, 2002 in Tokyo, Japan.',
        'Ado is a Japanese singer and Utaite who is often considered to be one of the most promising and popular young singers in Japan.',
        'Ado released her third album, "Ado\'s Utattemita Album," in December 2023, which is a cover album featuring various Vocaloid and J-pop songs that fans voted on for her to cover.',
        'On April 20, 2024, Ado announced her second studio album, "Zanmu."',
        'The album "Zanmu" was released on July 10, 2024.',
        'The release of "Zanmu" was accompanied by a live CD from Ado\'s first world tour "Wish," recorded at Peacock Theater, California.',
        'In 2023, Ado continued releasing singles, released her third album, and announced her first world tour.',
        'Notable singles released by Ado in 2023 include "Show" and "Kura Kura."',
        'The single "Show" quickly solidified itself as one of Ado\'s most popular songs.',
        'The single "Kura Kura" was used as the opening song for the anime "Spy x Family," becoming Ado\'s first TV anime opening.',
        'Ado is an Utaite born on October 24th, 2002, in Tokyo, Japan.',
        'Ado\'s major debut single, \'Usseewa\', was released in 2020 and became a social phenomenon.',
        'Ado\'s first album, \'Kyougen\', was released in 2022 and became a long-running hit.',
        'Ado captures the angst and cautious optimism of Japan\'s younger generations with her mesmerizing voice and powerful performances.',
        'Ado is a Japanese J-Pop singer responsible for some of the decade\'s biggest hits.',
        'In 2025, Ado embarked on their first world tour, named \'Wish.\'',
        'Ado\'s US shows during the \'Wish\' world tour sold out.',
        'Ado is a Japanese artist who continually seeks out challenges in her career.',
        'In 2024, Ado embarked on her first world tour, performing at sold-out venues across Asia, Europe, and North America.',
        'Ado became the first female artist to perform at the Japan National Stadium by hosting a pair of shows there.',
        'Starting in the summer of 2024, Ado completed a nationwide tour within Japan.',
        'Ado is actively engaging in collaborations to further expand her sound and showcase her confidence in songwriting.',
        'She announced her second world tour titled \'Hibana,\' which will include cities both familiar and new to her across Asia and Europe.',
        '\'Episode X\' is described as an important moment for Japanese music in the 2020s, featuring a collaboration between two significant creators and helping to spread Japan\'s music internationally.',
        'Ado\'s second world tour will begin in late April 2025, covering five continents and taking place in large arenas.',
        'The upcoming five-continent tour is noted as one of the biggest world tours ever undertaken by a J-pop artist, especially one still in her early 20s.',
        'Ado has released two major works listed on YouTube Music: \'Ado\'s Utattemita Album\' in 2023 and \'Ready For My Show Playlist\' in 2024.',
        'No other releases by Ado within the 2020-2025 timeframe are listed on the YouTube Music channel page.',
        'Ado will release a greatest hits album titled "Ado\'s Best Adobum" on April 9th, 2025.',
        '"Ado\'s Best Adobum" commemorates Ado\'s 5th debut anniversary.',
        'Pre-orders for "Ado\'s Best Adobum" are available at the Ado Official Music Shop.',
        'The album is promoted with the hashtag #AdosBestAdobum as stated by Ado Staff.',
        'Ado debuted as a singer in 2020.',
        'Her digital single "Usseewa" topped the Billboard Japan Hot 100 chart.',
        'The song "Usseewa" also reached number one on the Oricon Digital Singles Chart.',
        'Additionally, "Usseewa" topped the Oricon Streaming Chart.',
        'In 2022, Ado\'s song "New Genesis" was the theme song for the anime film One Piece Film: Red.',
        'The song "New Genesis" reached the top of Apple Music\'s Global Top 100 chart.',
        'In July 2024, Ado released her second studio album, Zanmu.',
        'The album Zanmu was preceded by 14 singles and debuted at number one on both the Oricon and Billboard Japan charts.',
        'Zanmu became Ado\'s fourth consecutive number one album on the Japan Hot Albums chart.',
        'Zanmu was Ado\'s second number one album on the Oricon Albums chart.',
        'In October 2024, Ado announced her second world tour Hibana, scheduled to take place from April to August 2025 across multiple continents.',
        'Imagine Dragons released a remix of their song \'Take Me to the Beach\' featuring Ado on December 16, 2024.',
        'In February 2025, it was announced that Ado would participate in the Nakamori Akina Tribute Album: Meikyo, covering Nakamori\'s 1984 single \'Jukkai\'.',
        'Ado featured as a vocalist in the digital single \'Shikabanēze\' by Jon-Yakitory, released on March 29, 2020.',
        'In May 2020, Ado contributed two songs, \'Call Boy\' by Syudou and \'Taema Naku Ai iro\' by Shishi Shishi, to Pony Canyon\'s compilation album Palette4.',
        'On October 15, 2020, Ado announced her debut with Universal Music Japan sublabel Virgin Music.',
        'In February 2025, Ado was appointed brand ambassador for Georgia, with her song \'Watashi ni Hanataba\' featured as the commercial theme song and her participating in the commercial as the narrator.',
        'Ado\'s music video released on her YouTube channel reached 5 million views by November 14, 2020.',
        'On December 10, 2020, Ado\'s song \'Usseewa\' ranked number 1 on Spotify Viral 50 Japan.',
        'Ado released her second single \'Readymade\', written by Vocaloid producer Surii, as a digital release on December 24, 2020.',
        'Having not shared any personal information of herself, little is known about Ado.',
        '"Stay Gold" is the third ending theme song for BEYBLADE X, set to be released on May 16, 2025, and is a collaboration between Ado and music producer and DJ Jax Jones.',
        '"New Genesis" made history by becoming the first Japanese song to reach the #1 position on the Apple Music Global 100 playlist.',
        'On the Billboard charts, "New Genesis" climbed to the top 20 on the Global 200 ranking.',
        'The song "I\'m invincible" (私は最強, Watashi wa Saikyou) is the first insert song from the anime film ONE PIECE FILM RED, released as a digital single on June 22, 2022.',
        '"Fleeting Lullaby" earned a Platinum certification in Streaming by the Recording Industry Association of Japan (RIAJ) in April 2023.',
        '"Fleeting Lullaby" also surpassed 100 million streams on the BILLBOARD JAPAN charts.',
        '"Tot Musica" is the fourth insert song from ONE PIECE FILM RED, released on August 10, 2022.',
        '"New Genesis" (新時代, Shinjidai) is the opening theme song for ONE PIECE FILM RED and was released as a digital single on June 8, 2022.',
        '"New Genesis" was composed by the world-renowned electronic music producer, Yasutaka Nakata.',
        'These achievements highlight the song\'s global impact, bringing international recognition to Ado\'s name.',
        'Ado has two major releases listed on YouTube Music\'s page: \'Ado\'s Utattemita Album\' (2023) and \'Ready For My Show Playlist\' (2024).',
        'No other releases by Ado within the 2020-2025 timeframe are listed on the YouTube Music channel page.',
        'Ado\'s major release between 2020 and 2025 is the album Zanmu.',
        'The album Zanmu was released in conjunction with her 2025 world tour called "Ado WORLD TOUR 2025 "Hibana"".',
        'There are no details on any other releases by Ado between 2020 and 2025 provided in the article.',
        'Ado\'s major releases from 2020-2025 include two studio albums titled Kyōgen (2022) and Zanmu (2024).',
        'Ado released a soundtrack album called Uta\'s Songs: One Piece Film Red in 2022.',
        'In 2023, Ado released a cover album named Ado\'s Utattemita Album.',
        'Ado\'s greatest hits album, titled Ado\'s Best Adobum, was released on April 9, 2025.',
        'The album Ado\'s Best Adobum includes songs from Ado\'s previous albums and singles such as Shoka, Sakura Biyori and Time Machine, Episode X, and Elf.',
        'The singles Shoka, Sakura Biyori and Time Machine, Episode X, and Elf were released between October 2024 and January 2025.',
        'Ado released a greatest hits album called *Ado\'s Best Adobum* in April 2025.',
        'The album celebrates the 5th anniversary of Ado\'s major-label debut with the single "Usseewa" in 2020.',
        'The release of the greatest hits album coincided with a new music video for her single "ROCKSTAR."',
        'The single "ROCKSTAR" serves as the theme song for Marubeni\'s new corporate campaign.',
        'Ado\'s first best album, titled "Ado no Best Adabumu," will be released on April 9, 2025.',
        'The album celebrates Ado\'s fifth anniversary since her major debut in October 2020 with the song "Usseewa."',
        'The two-CD album includes 40 tracks, featuring major hits such as "Usseewa," "Giragira," "Odori," "Shinjidai," and "Shou."',
        'The album contains two unreleased songs and the previously unreleased track "Hello Signals."',
        'There are six different versions of the album available, some of which include a Blu-ray/DVD of Ado\'s first solo concert.',
        'Ado has planned a world tour running from April to August 2025.',
        'Ado\'s career details up to 2022 include her debut single \'Usseewa\'.',
        '\'New Genesis\' was used as the theme song for One Piece Film: Red.',
        'Ado is a J-Pop Vocaloid star wh success with the song "Usseewa," which reached #1 on several Japanese charts.',
        '"Usseewa" became the youngest artist to hit 100 million plays in 17 weeks.',
        'The article does not list any specific awards or recognitions received by Ado.',
        'Ado won the Artist of the Year award at the Billboard Japan Music Awards in 2022.',
        'Ado\'s song "New Genesis" won Song of the Year at the MTV Video Music Awards Japan.',
        'At the 64th Japan Record Awards, Ado won the Excellent Work Awards and a Special Prize.',
        'Ado was nominated for Best Anime Song for "New Genesis" at the 7th Crunchyroll Anime Awards in 2023.',
        'Ado won the Excellent Work Awards for the song "Show" at the 65th Japan Record Awards.',
        'Ado won Best New Asian Artist (Japan) at the Mnet Asian Music Awards in 2021.',
        'Ado\'s song "Usseewa" won MTV Breakthrough Song at the MTV Video Music Awards Japan.',
        'Ado received a Special Award at the 63rd Japan Record Awards.',
        'Ado\'s song "Show" achieved a certification of 3× Platinum from the Recording Industry Association of Japan (RIAJ).',
        'Ado\'s song "Kura Kura" was certified Gold by the Recording Industry Association of Japan (RIAJ).',
        'Ado is a J-Pop Vocaloid star whose career is highlighted by immense success.',
        'Ado\'s song \'Usseewa\' reached number one on several Japanese music charts.',
        '\'Usseewa\' became the youngest artist\'s track to achieve 100 million plays within 17 weeks.',
        'The article does not list any specific awards or recognitions received by Ado as of July 2024.',
        'Ado\'s career details are covered up to the year 2022.',
        'Ado debuted with the single "Usseewa."',
        'Her song "New Genesis" was used as the theme song for One Piece Film: Red.',
        'The provided text does not contain information on awards and recognitions received by Ado as of 2025.',
        'Ado\'s musical style blends pop and rock, characterized by an explosive dynamic vocal range and raw emotion.',
        'Ado has been active in her musical career from at least 2020 to 2025.',
        'Ado is set to release a greatest hits album in 2025 titled Ado\'s Best Adobum.',
        'The 2025 greatest hits album will offer a retrospective of Ado\'s musical evolution from 2020 to 2025.',
        'The text does not detail specific genre shifts in Ado\'s music in 2025.',
        'Ado is a Japanese J-Pop artist who started gaining popularity in 2020.',
        'Her breakout hit "Usseewa," which went viral in 2020, features a blend of controlled rage, sneers, and screams, reflecting a rejection of societal conformity.',
        'Ado\'s music blends elements of electronic music with powerful, emotive vocals.',
        'Her music often falls under the Vocaloid subculture, a genre utilizing vocal synthesizing software.',
        'Her trajectory suggests a continued exploration of powerful vocals and electronically driven sounds within the J-Pop landscape.',
        'Her 2024 album Zanmu shows a possibly more mature perspective in her music.',
        'Japanese pop star Ado\'s music was launched into low Earth orbit as part of the BandWagon2 project, which serves as a prelude to sending it to the Moon.',
        'Ado is identified as a "Japanese pop sensation," indicating her genre is likely J-Pop.',
        'The article does not provide specific details about Ado\'s musical style in 2025 and suggests that more detailed information would require consulting other sources.',
        'Ado\'s concert tour starts on April 27, 2025, and ends on August 24, 2025.',
        'Ado will perform in 22 cities during the 2025 tour.',
        'Ado\'s most recent concert was held in  さいたま市 at さいたまスーパーアリーナ.',
        'Ado experienced success with a sold-out tour earlier in the year prior to her 2025 world tour.',
        'Ado will embark on her second world tour titled \'Hibana\' in 2025, which follows her first world tour named \'Wish\'.',
        'The \'Hibana\' world tour reflects Ado\'s artistic growth since the \'Wish\' tour and aims to feature her best performances yet.',
        'Ado stated, \'If my first world tour embodied my \'Wish,\' then my second will ignite the spark I want to light in the world.\'',
        'The name of the 2025 world tour, \'Hibana,\' means \'spark\' in Japanese, chosen to carry her heritage on this journey.',
        'Ado has 25 upcoming shows scheduled for the years 2024 and 2025.',
        'Upcoming tour dates for Ado in 2024 include performances on March 16th at the Peacock Theater in Los Angeles, California, April 27th at Saitama Super Arena in Saitama, Japan, May 25th at Qudos Bank Arena in Sydney, Australia, May 27th at Rod Laver Arena in Melbourne, Australia, June 10th at Sportpaleis in Antwerp, Belgium, and June 14th at Royal Arena in Copenhagen, Denmark.',
        'Ado\'s 2024 activities included a sold-out world tour called "Wish."',
        'Ado was the first female artist to perform live at Japan National Stadium.',
        'In 2024, Ado conducted the "Profile of Mona Lisa" Japan tour.',
        'Ado released the album *Zanmu* in July 2024.',
        'Ado appeared at the Expo 2025 Osaka Kansai opening ceremony in April 2025.',
        'Ado has embarked on the Ado WORLD TOUR 2025 "Hibana," which is one of the largest world tours ever by a Japanese artist.',
        'The Ado WORLD TOUR 2025 "Hibana" includes a show at the SAP Center in San Jose on July 13, 2025.',
        'Ado has 25 upcoming shows in 2024 and 2025.',
        'Upcoming tour dates include March 16th, 2024 at the Peacock Theater in Los Angeles, CA.',
        'Ado will perform on April 27th, 2024 at Saitama Super Arena in Saitama, Japan.',
        'There is a scheduled show on May 25th, 2024 at Qudos Bank Arena in Sydney, Australia.',
        'Ado has a concert planned for May 27th, 2024 at Rod Laver Arena in Melbourne, Australia.',
        'On June 10th, 2024, Ado will perform at Sportpaleis in Antwerp, Belgium.',
        'On June 14th, 2024, Ado will perform at Royal Arena in Copenhagen, Denmark.',
        'Ado\'s 2025 World Tour, titled "Hibana," will run from April 26th to August 24th, 2025.',
        'The tour will include over 34 concerts across Asia, Europe, North America, and Latin America.',
        'Major cities on the tour include Seoul, London, Paris, and Los Angeles.',
        'The 2025 World Tour is in partnership with Crunchyroll.',
        'The tour will feature songs from Ado\'s new album, Zanmu.',
        'Concert lengths are estimated to be between 60 and 120 minutes based on previous tour feedback.',
        'No opening acts have been announced for the 2025 World Tour.',
        'The article does not provide any information about Ado\'s activities or tours in 2024.',
        'Ado\'s tour starts on April 27, 2025 and ends on August 24, 2025.',
        'Ado will play in 22 cities during the tour.',
        'Ado\'s most recent concert was held in さいたま市 at さいたまスーパーアリーナ.',
        'Ado is a Japanese anime pop star whose animated voice is rattling the music industry.',
        'Her new album titled \'Zanmu\' has been recently released.',
        'Ado is planning a 2025 world tour, which is described as her most ambitious project to date.',
        'No opening acts have been announced yet for Ado\'s 2025 tour.',
        'Ado\'s Tour setlist may change depending on the concert venue and dates.',
        'Ado\'s 2025 tour merch will be available both at the venue shop and online through different platforms.',
        'Rumors suggest that Ado might have secured a ticket to perform at Coachella 2025.',
        'Ado\'s world tour is anticipated to be so massive that it will leave audiences speechless.',
    ]

    grouped_statements = group_similar_statements(test_statements)
    print(grouped_statements)
