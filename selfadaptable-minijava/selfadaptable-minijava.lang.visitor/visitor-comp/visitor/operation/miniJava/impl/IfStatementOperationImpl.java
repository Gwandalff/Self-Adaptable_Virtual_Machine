package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.IfStatement;
import miniJava.visitor.miniJava.State;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BlockOperation;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.IfStatementOperation;

public class IfStatementOperationImpl extends StatementOperationImpl implements IfStatementOperation {
	private final IfStatement it;

	private final VisitorInterface vis;

	public IfStatementOperationImpl(IfStatement it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatement(State state) {
		BooleanValue booleanValue = ((BooleanValue) (((BooleanValue) (((ExpressionOperation)this.it.getExpression().accept(vis)).evaluateExpression((State) (state))))));
		if(booleanValue.isValue()) {
			((BlockOperation)this.it.getThenBlock().accept(vis)).evaluateStatement((State) (state));
		}
		else {
			if(!EqualService.equals((this.it.getElseBlock()), (null))) {
				((BlockOperation)this.it.getElseBlock().accept(vis)).evaluateStatement((State) (state));
			}
		}
	}
}
