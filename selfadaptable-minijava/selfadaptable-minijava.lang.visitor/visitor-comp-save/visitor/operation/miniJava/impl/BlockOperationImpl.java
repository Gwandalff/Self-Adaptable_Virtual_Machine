package visitor.operation.miniJava.impl;

import java.lang.Integer;
import miniJava.visitor.miniJava.Block;
import miniJava.visitor.miniJava.Frame;
import miniJava.visitor.miniJava.State;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.BlockOperation;
import visitor.operation.miniJava.StateOperation;
import visitor.operation.miniJava.StatementOperation;

public class BlockOperationImpl extends StatementOperationImpl implements BlockOperation {
	private final Block it;

	private final VisitorInterface vis;

	public BlockOperationImpl(Block it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}

	public void evaluateStatementKeepContext(State state) {
		((StateOperation)state.accept(vis)).pushNewContext();
		Frame currentFrame = ((Frame) (((StateOperation)state.accept(vis)).findCurrentFrame()));
		int lgt = ((int) (CollectionService.size(this.it.getStatements())));
		int i = ((int) (0));
		while ((((i) < (lgt)) && (EqualService.equals((currentFrame.getReturnValue()), (null))))) {
			((StatementOperation)CollectionService.get(this.it.getStatements(), i).accept(vis)).evaluateStatement((State) (state));
			i = ((Integer) ((i) + (1)));
		}
	}

	public void evaluateStatement(State state) {
		((BlockOperation)this.it.accept(vis)).evaluateStatementKeepContext((State) (state));
		((StateOperation)state.accept(vis)).popCurrentContext();
	}
}
