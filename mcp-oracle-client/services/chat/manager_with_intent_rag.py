# services/chat/manager_with_intent.py
"""Modified chat manager with minimal Intent Agent integration."""
import logging
from typing import Optional, Tuple, List, Dict, Any
import json
from models.chat import ChatResponse, ConversationContext
from services.chat.manager import ChatManager
from services.chat.intent_agent import IntentAgent, SemanticIntent
from services.sql_converter import sql_converter
from services.query_executor import query_executor
from services.mcp_client import mcp_client
from services.rag_agent.graph_context_retriever import graph_rag_retriever
from services.rag_agent.rag_system import RAGSystem

logger = logging.getLogger(__name__)


class ChatManagerWithIntent(ChatManager):
    """
    Extended ChatManager that adds Intent Agent capabilities
    while preserving all existing functionality.
    """
    
    def __init__(self, rag_config):
        """Initialize with Intent Agent added to existing components."""
        super().__init__()
        self.intent_agent = IntentAgent()
        self.rag_config = rag_config
        # Initialize components
        self.rag_system = RAGSystem(
            persist_directory=rag_config.persist_directory,
            collection_prefix=rag_config.collection_name,
            embedding_model=rag_config.embedding_model
        )
        logger.info("Chat manager initialized with Intent Agent extension")
    
    async def initialize_with_schema(self, data_dictionary: Dict[str, Dict[str, Any]]) -> None:
        """Initialize RAG system with schema information."""
        # Build schema graph
        self.rag_system.build_schema_graph(data_dictionary)
        
        # Train with DDL and dictionary
        for table_name, table_info in data_dictionary.items():
            # Add to schema collection
            doc_text = f"Table: {table_name}\n"
            doc_text += f"Description: {table_info.get('description', '')}\n"
            doc_text += f"Columns: {json.dumps(table_info.get('columns', {}))}\n"
            
            self.rag_system.collections['schema'].add(
                documents=[doc_text],
                metadatas=[{
                    'table_name': table_name,
                    'column_count': len(table_info.get('columns', {})),
                    'has_relationships': len(table_info.get('relationships', {})) > 0
                }],
                ids=[f"schema_{table_name}"]
            )
        
        logger.info(f"Initialized with {len(data_dictionary)} tables")

    async def process_message(self, session_id: str, message: str) -> ChatResponse:
        """
        Process message with optional semantic intent extraction.
        Falls back to original behavior if intent extraction fails.
        """
        # Get the context
        context = self.context_manager.ensure_session(session_id)
        
        # Add user message to history (as before)
        self.context_manager.add_message(session_id, "user", message)
        
        # Classify intent (existing functionality)
        intent = await self.intent_classifier.classify_intent(message, context)
        logger.info(f"Processing message with intent: {intent}")
        
        # Try to extract semantic intent for new queries only
        semantic_intent = None
        if intent in ["new_query", "execute_immediately"]:
            try:
                semantic_intent = await self.intent_agent.extract_intent(
                    message,
                    {"last_sql": context.last_sql} if context.last_sql else None
                )
                logger.info(f"Semantic intent extracted: {semantic_intent.to_dict()}")
                
                # Store in context for reference
                if hasattr(context, '__dict__'):
                    context.__dict__['last_semantic_intent'] = semantic_intent.to_dict()
                    
            except Exception as e:
                logger.warning(f"Semantic intent extraction failed, continuing normally: {e}")
                # Continue with normal flow
        
        # Use the original routing logic
        if intent == "follow_up":
            response = await self._handle_follow_up(session_id, message, context)
        elif intent == "modify_query":
            response = await self._handle_modify_query(session_id, message, context)
        elif intent == "execute_immediately":
            response = await self._handle_execute_immediately(session_id, message, context)
        else:  # new_query
            # Try enhanced query generation if we have semantic intent
            if semantic_intent:
                response = await self._handle_new_query_with_intent_rag(
                    session_id, message, context, semantic_intent
                )
            else:
                response = await self._handle_new_query(session_id, message, context)
        
        # Add assistant response to history (as before)
        self.context_manager.add_message(
            session_id, 
            "assistant", 
            response.message,
            {"action": response.action_type, "sql_query": response.sql_query}
        )
        
        # Update context if query was executed (as before)
        if response.result and response.sql_query:
            self.context_manager.update_last_query(
                session_id,
                response.sql_query,
                response.result
            )
        elif response.sql_query and response.requires_confirmation:
            context.last_sql = response.sql_query
        
        return response
    
    async def _handle_new_query_with_intent_rag(self, session_id, message, context, semantic_intent):
        try:
            # Get RAG context from intent
            rag_context = None
            if semantic_intent:
                intent_dict = semantic_intent.to_dict()
                #rag_context = await graph_rag_retriever.retrieve_context(intent_dict, k=3, depth=1)
                rag_context = self.rag_system.get_relevant_context(message)
                #logger.info(f"RAG retrieved tables: {rag_context.primary_tables}")
            
            # Continue with existing SQL generation
            response = await self.query_handler.generate_query(message, intent_dict ,rag_context, session_id)
            
            # Optional: Add context info to confirmation message
            # if response.requires_confirmation and rag_context:
            #     # Add tables being used to the message
            #     tables_info = f"\n**Using Tables:** {', '.join(rag_context.primary_tables)}\n"
            #     response.message = response.message.replace("```sql", f"{tables_info}\n```sql")
            
            #Enhance the confirmation message with intent understanding
            if response.requires_confirmation and semantic_intent:
                intent_summary = self._build_intent_summary(semantic_intent)
                if intent_summary:
                    # Insert intent summary before SQL
                    original_msg = response.message
                    sql_start = original_msg.find("```sql")
                    if sql_start > 0:
                        response.message = (
                            original_msg[:sql_start] + 
                            f"\n{intent_summary}\n\n" + 
                            original_msg[sql_start:]
                        )
            

            return response
            
        except Exception as e:
            logger.error(f"RAG context failed: {e}")
            # Fallback to existing flow
            return await self._handle_new_query(session_id, message, context)

    # async def _handle_new_query_with_intent(self, session_id, message, context, semantic_intent):
    #     try:
    #         # Get RAG context from intent
    #         rag_context = None
    #         if semantic_intent:
    #             intent_dict = semantic_intent.to_dict()
    #             rag_context = await graph_rag_retriever.retrieve_context(intent_dict, k=3, depth=1)
    #             logger.info(f"RAG retrieved tables: {rag_context.primary_tables}")
            
    #         # Continue with existing SQL generation
    #         response = await self.query_handler.generate_query(message, intent_dict ,rag_context, session_id)
            
    #         # Optional: Add context info to confirmation message
    #         # if response.requires_confirmation and rag_context:
    #         #     # Add tables being used to the message
    #         #     tables_info = f"\n**Using Tables:** {', '.join(rag_context.primary_tables)}\n"
    #         #     response.message = response.message.replace("```sql", f"{tables_info}\n```sql")
            
    #         #Enhance the confirmation message with intent understanding
    #         if response.requires_confirmation and semantic_intent:
    #             intent_summary = self._build_intent_summary(semantic_intent)
    #             if intent_summary:
    #                 # Insert intent summary before SQL
    #                 original_msg = response.message
    #                 sql_start = original_msg.find("```sql")
    #                 if sql_start > 0:
    #                     response.message = (
    #                         original_msg[:sql_start] + 
    #                         f"\n{intent_summary}\n\n" + 
    #                         original_msg[sql_start:]
    #                     )
            

    #         return response
            
    #     except Exception as e:
    #         logger.error(f"RAG context failed: {e}")
    #         # Fallback to existing flow
    #         return await self._handle_new_query(session_id, message, context)


    def _build_intent_summary(self, intent: SemanticIntent) -> str:
        """Build a brief summary of what was understood from the query."""
        parts = []
        
        if intent.primary_intent:
            parts.append(f"**Understood:** {intent.primary_intent.replace('_', ' ').title()} query")
        
        if intent.metrics:
            parts.append(f"**Analyzing:** {', '.join(intent.metrics)}")
        
        if intent.dimensions:
            parts.append(f"**Grouped by:** {', '.join(intent.dimensions)}")
        
        if intent.filters:
            parts.append(f"**With filters:** {', '.join(intent.filters[:3])}")  # Limit to 3
        
        return "\n".join(parts) if parts else ""
    
    def get_last_semantic_intent(self, session_id: str) -> Optional[dict]:
        """Get the last semantic intent for a session."""
        context = self.get_context(session_id)
        if context and hasattr(context, '__dict__'):
            return context.__dict__.get('last_semantic_intent')
        return None
