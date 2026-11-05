/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_test.h"
#include "util/half_float.h"

/* Refused shape: roundeven d32 <- s16. Copy propagation folds a widening
 * source move into the rounding instruction, but G17P's rounding forms have
 * width-matched operands. Cover the exact instruction and sign modifiers. */
TEST(AGX3UnaryLowering, RoundevenHalfSourceToWordResult)
{
   void *mem = ralloc_context(nullptr);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (auto stage : {MESA_SHADER_COMPUTE, MESA_SHADER_FRAGMENT}) {
      for (unsigned signs = 0; signs < 4; ++signs) {
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         b->shader->stage = stage;
         agx_index half = agx_temp(b->shader, AGX_SIZE_16);
         half.abs = signs & 1;
         half.neg = signs & 2;
         agx_instr *round = agx_roundeven_to(
            b, agx_temp(b->shader, AGX_SIZE_32), half);
         agx_lower_agx3_mixed_width_unary(b->shader);
         ASSERT_EQ(round->src[0].size, AGX_SIZE_32);
         EXPECT_FALSE(round->src[0].abs);
         EXPECT_FALSE(round->src[0].neg);
         agx_instr *move = list_entry(round->link.prev, agx_instr, link);
         EXPECT_EQ(move->op, AGX_OPCODE_FADD);
         EXPECT_EQ(move->src[0].size, AGX_SIZE_16);
         EXPECT_EQ(move->src[0].abs, bool(signs & 1));
         EXPECT_EQ(move->src[0].neg, bool(signs & 2));
         EXPECT_TRUE(agx_is_equiv(move->dest[0], round->src[0]));
      }
   }
   ralloc_free(mem);
}

TEST(AGX3UnaryLowering, PreservesMatchedWidthsAndOtherGenerations)
{
   void *mem = ralloc_context(nullptr);
   for (auto encoding : {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3B}) {
      agx_shader_key key = {};
      key.dev.encoding = encoding;
      for (auto size : {AGX_SIZE_16, AGX_SIZE_32}) {
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         agx_index source = agx_temp(b->shader, size);
         agx_instr *round = agx_roundeven_to(b, agx_temp(b->shader, size), source);
         agx_lower_agx3_mixed_width_unary(b->shader);
         EXPECT_TRUE(agx_is_equiv(round->src[0], source));
      }
   }
   ralloc_free(mem);
}

TEST(AGX3UnaryLowering, M5KeepsNarrowingAfterFullUnary)
{
   for (auto op : {AGX_OPCODE_FLOOR, AGX_OPCODE_CEIL, AGX_OPCODE_TRUNC,
                   AGX_OPCODE_ROUNDEVEN, AGX_OPCODE_LOG2, AGX_OPCODE_EXP2,
                   AGX_OPCODE_RSQRT, AGX_OPCODE_SRSQRT, AGX_OPCODE_DFDX,
                   AGX_OPCODE_DFDY, AGX_OPCODE_SINC_AGX3}) {
      for (auto size : {AGX_SIZE_16, AGX_SIZE_32}) {
         void *mem = ralloc_context(nullptr);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         b->shader->key = &key;
         agx_index input = agx_preload(b, agx_register(0, size));
         agx_instr *I = agx_alloc_instr(b, op, 1, 1);
         I->src[0] = input;
         I->dest[0] = agx_temp(b->shader, size);
         agx_builder_insert(&b->cursor, I);
         agx_index result = agx_temp(b->shader, AGX_SIZE_16);
         agx_fmov_to(b, result, I->dest[0]);
         agx_unit_test(b, result);
         agx_optimizer_backward(b->shader);
         EXPECT_EQ(I->dest[0].size, size) << op;
         ralloc_free(mem);
      }
   }
}

TEST(AGX3UnaryLowering, M5SaturatedHalfImmediates)
{
   for (auto op : {AGX_OPCODE_HADD, AGX_OPCODE_HFMA}) {
      for (unsigned literal : {0u, 0x80u, 0x38u}) {
         void *mem = ralloc_context(nullptr);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         b->shader->key = &key;
         agx_instr *I = agx_alloc_instr(b, op, 1, op == AGX_OPCODE_HFMA ? 3 : 2);
         I->dest[0] = agx_temp(b->shader, AGX_SIZE_16);
         for (unsigned s = 0; s < I->nr_srcs; ++s)
            I->src[s] = agx_temp(b->shader, AGX_SIZE_16);
         I->src[1] = agx_immediate(literal);
         I->saturate = true;
         agx_builder_insert(&b->cursor, I);
         agx_lower_uniform_sources(b->shader);
         EXPECT_TRUE(I->saturate);
         EXPECT_EQ(I->src[1].size, AGX_SIZE_16);
         EXPECT_EQ(I->src[1].type, AGX_INDEX_NORMAL);
         agx_instr *move = list_entry(I->link.prev, agx_instr, link);
         EXPECT_EQ(move->op, AGX_OPCODE_MOV_IMM);
         EXPECT_EQ(move->imm, _mesa_float_to_half(agx_minifloat_decode(literal)));
         ralloc_free(mem);
      }
   }
}
